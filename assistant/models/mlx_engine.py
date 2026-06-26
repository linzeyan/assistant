"""In-memory engine pool for the native MLX backend.

This is the project's own version of omlx's ``EnginePool`` — the "preload / unload /
switch" management the user wants, owned natively so it needs no external omlx
server. The pool keeps loaded ``(model, tokenizer)`` engines in memory, evicts the
least-recently-used one when over budget, supports manual load/unload, and can pin
models so a hot model is never evicted.

Loading and generation are blocking (MLX runs on the CPU/GPU synchronously), so the
pool loads via ``asyncio.to_thread`` and callers stream generation off the event
loop (see ``MlxModelService``). The actual ``mlx_lm`` import is deferred to the
default loader so this module imports fine on machines without MLX installed, and so
tests can inject a fake loader.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
from collections import OrderedDict
from collections.abc import Callable, Iterator
from pathlib import Path

log = logging.getLogger("assistant")


def _release_mlx_memory() -> None:
    """Return a just-unloaded model's unified memory to the system.

    Dropping the last Python reference to an mlx model is necessary but not sufficient:
    MLX keeps freed buffers in a Metal allocator cache, so resident memory doesn't fall
    until that cache is cleared. A GC pass finalises the now-unreferenced model, then
    ``mx.clear_cache()`` releases the pooled buffers. Import-guarded so it's a no-op on
    machines without MLX (and under the fake-loader tests), and tolerant of the older
    ``mx.metal.clear_cache`` spelling.
    """
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None
    )
    if callable(clear):
        clear()
    # Log the active/cache footprint so "unload didn't free memory" reports are diagnosable:
    # active = memory still held by live arrays (a leaked reference shows up here), cache =
    # pooled-but-free buffers (should drop to ~0 right after clear_cache).
    active = getattr(mx, "get_active_memory", None)
    cache = getattr(mx, "get_cache_memory", None)
    if callable(active) and callable(cache):
        log.info(
            "mlx memory after release: active=%.2fGB cache=%.2fGB",
            active() / 1e9, cache() / 1e9,
        )


def _messages_for_template(messages: list[dict]) -> list[dict]:
    """Return messages with assistant tool_calls' ``arguments`` parsed from JSON string to a
    dict, for chat-template rendering only.

    Sessions persist tool_calls in OpenAI wire format — ``arguments`` is a JSON *string* — but
    HF chat templates expect a parsed mapping: Qwen3.x iterates it with jinja ``| items``,
    which raises "Can only get item pairs from a mapping" on a string (the real cause of the
    "web search just fails" reports). Copy-on-write so the persisted history stays
    string-typed; a value that doesn't parse is left untouched.
    """
    out: list[dict] = []
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m)
            continue
        new_tcs = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    new_tcs.append(tc)  # not JSON — leave as-is, template may accept strings
                    continue
                new_tcs.append({**tc, "function": {**fn, "arguments": parsed}})
            else:
                new_tcs.append(tc)
        out.append({**m, "tool_calls": new_tcs})
    return out


def _render_prompt(templater, messages: list[dict], tools: list[dict] | None) -> str:
    """Render the chat prompt, normalising tool_calls first and falling back ONLY when the
    tokenizer genuinely rejects the ``tools`` kwarg.

    A TypeError from *inside* the template (a message-shape mismatch) must surface — retrying
    without tools would just fail the same way and mask the real cause. The previous blanket
    ``except TypeError`` swallowed exactly that, hiding the Qwen3.x tool_calls render bug.
    """
    messages = _messages_for_template(messages)
    try:
        return templater(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    except TypeError as exc:
        if "tools" not in str(exc):  # not the "template doesn't accept tools" case — surface it
            raise
        return templater(messages, add_generation_prompt=True, tokenize=False)


class MlxEngine:
    """A loaded mlx-lm model + tokenizer that streams text for one chat turn."""

    def __init__(self, model: object, tokenizer: object):
        self._model = model
        self._tokenizer = tokenizer

    def stream_text(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        **_ignored,
    ) -> Iterator[str]:
        # Imported lazily: MLX is a heavy, Apple-Silicon-only dependency.
        from mlx_lm import stream_generate

        # Passing tools lets the chat template render them in the model's expected
        # tool-calling format. _render_prompt normalises tool_calls and handles the
        # tools-kwarg fallback without masking real template errors.
        prompt = _render_prompt(self._tokenizer.apply_chat_template, messages, tools)
        for response in stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=max_tokens
        ):
            text = getattr(response, "text", None)
            if text:
                yield text


class VlmChatEngine:
    """A loaded mlx-vlm model used as a *text* chat model.

    Omni / vision-language checkpoints (e.g. Qwen-VL) carry a ``vision_config`` and
    mlx-lm can't load them — only mlx-vlm can. This wraps mlx-vlm so such a model can
    still be the chat model (text in, text out), mirroring ``MlxEngine``'s contract.
    Image *input* isn't wired into chat here; the agent reads images via the separate
    ``view_image`` tool. The processor proxies the model's chat template, so multi-turn
    role handling matches the LLM path.
    """

    def __init__(self, model: object, processor: object):
        self._model = model
        self._processor = processor

    def stream_text(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        **_ignored,
    ) -> Iterator[str]:
        from mlx_vlm import stream_generate

        templater = getattr(self._processor, "apply_chat_template", None) or getattr(
            self._processor, "tokenizer"
        ).apply_chat_template
        prompt = _render_prompt(templater, messages, tools)
        for chunk in stream_generate(
            self._model, self._processor, prompt, max_tokens=max_tokens
        ):
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, str):
                text = chunk
            if text:
                yield text


def _load_llm(path: Path) -> MlxEngine:
    from mlx_lm import load

    try:
        model, tokenizer = load(str(path))
    except Exception as exc:
        # mlx-lm raises a bare ``ValueError("Model type <x> not supported.")`` when the
        # installed mlx-lm predates a model's architecture (e.g. qwen3_5 / qwen3_6). The
        # checkpoint is fine — the venv's mlx-lm is just too old. Re-raise with the version
        # and a pointer to the in-app updater instead of a cryptic message (N11).
        msg = str(exc)
        if "not supported" in msg.lower() and "model type" in msg.lower():
            from importlib import metadata

            try:
                installed = metadata.version("mlx-lm")
            except Exception:
                installed = "unknown"
            raise RuntimeError(
                f"{msg.rstrip('.')}. Installed mlx-lm is {installed}, too old for this "
                "model's architecture — update it in Settings ▸ Managed tools (更新套件), "
                "then restart the backend."
            ) from exc
        raise
    return MlxEngine(model, tokenizer)


def _load_vlm(path: Path) -> VlmChatEngine:
    try:
        from mlx_vlm import load
    except ImportError as exc:  # vlm models need the optional extra
        raise RuntimeError(
            'this is a vision-language model; install mlx-vlm to chat with it: '
            'uv pip install -e ".[vlm]"'
        ) from exc
    model, processor = load(str(path))
    return VlmChatEngine(model, processor)


def _default_loader(path: Path) -> object:
    # Dispatch by model family: a VL/omni checkpoint loads through mlx-vlm, everything
    # else through mlx-lm. classify_kind re-reads config.json from the path, so the pool
    # stays generic (loader takes only a path) and test loaders need no changes.
    from .mlx_discovery import classify_kind

    return _load_vlm(path) if classify_kind(path) == "vlm" else _load_llm(path)


class MlxEnginePool:
    """LRU pool of loaded engines. ``loader`` is injectable for tests."""

    def __init__(
        self,
        *,
        max_loaded: int = 1,
        loader: Callable[[Path], object] | None = None,
        pinned: set[str] | None = None,
    ):
        self._max = max(1, max_loaded)
        self._loader = loader or _default_loader
        # Insertion order == LRU order; move_to_end marks most-recently-used.
        self._loaded: OrderedDict[str, object] = OrderedDict()
        self._pinned: set[str] = set(pinned or set())
        self._lock = asyncio.Lock()

    async def acquire(self, model_id: str, path: Path) -> object:
        """Return the loaded engine for ``model_id``, loading (and evicting) as needed."""
        async with self._lock:
            if model_id in self._loaded:
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            self._evict(exclude=model_id)
            # Load outside would race the lock; loading under the lock serialises
            # model switches, which is what we want (one heavy load at a time).
            log.info("loading model into pool: %s", model_id)
            engine = await asyncio.to_thread(self._loader, path)
            self._loaded[model_id] = engine
            self._loaded.move_to_end(model_id)
            log.info("model loaded: %s (pool now: %s)", model_id, self.loaded_ids())
            return engine

    def _evict(self, exclude: str) -> None:
        # Drop LRU non-pinned engines until under budget. Dropping the last reference is
        # necessary but not sufficient to free MLX's unified memory — _release_mlx_memory
        # (called once after any eviction) clears the Metal buffer cache too.
        evicted = False
        while len(self._loaded) >= self._max:
            victim = next(
                (k for k in self._loaded if k != exclude and k not in self._pinned),
                None,
            )
            if victim is None:
                break  # everything left is pinned — exceed budget rather than evict it
            self._loaded.pop(victim)
            log.info("evicted LRU model from pool: %s (making room for %s)", victim, exclude)
            evicted = True
        if evicted:
            _release_mlx_memory()

    async def load(self, model_id: str, path: Path) -> None:
        await self.acquire(model_id, path)

    async def unload(self, model_id: str) -> bool:
        async with self._lock:
            removed = self._loaded.pop(model_id, None) is not None
            if removed:
                log.info("unloading model from pool: %s (pool now: %s)", model_id, self.loaded_ids())
                _release_mlx_memory()
            else:
                log.info("unload requested for model not in pool: %s", model_id)
            return removed

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def loaded_ids(self) -> list[str]:
        return list(self._loaded.keys())

    def pin(self, model_id: str) -> None:
        self._pinned.add(model_id)

    def unpin(self, model_id: str) -> None:
        self._pinned.discard(model_id)
