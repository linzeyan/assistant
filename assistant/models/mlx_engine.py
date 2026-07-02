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
import os
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
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


def _total_ram_bytes() -> int | None:
    """Physical RAM in bytes, or None if it can't be determined. Used to default the pool's
    admission ceiling to the machine's actual memory so an oversized model fails loud with a clear
    message instead of OOM-crashing the backend — "check the resource fits before loading"."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")  # macOS + Linux
    except (ValueError, OSError, AttributeError):
        return None


def _active_memory_bytes() -> int | None:
    """Current MLX unified-memory footprint in bytes, or None when MLX isn't importable
    (fake-loader tests / non-MLX machines). Lets the pool refine a model's disk-size estimate
    with its real post-load footprint for the admission ceiling (see ``MlxEnginePool.acquire``)."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    active = getattr(mx, "get_active_memory", None)
    return int(active()) if callable(active) else None


def _estimate_model_bytes(path: Path) -> int:
    """Estimate a model's resident footprint from its on-disk weight shards.

    The summed ``*.safetensors`` size is the dominant term (the weights) and the only signal
    available *before* a load. It UNDER-estimates — runtime adds a KV cache and activations that
    grow with context — so it's used only as the pre-load admission guess and is replaced by the
    measured active-memory delta once the model is in. Returns 0 when no weight files are found,
    which disables byte-admission for that model rather than guessing from nothing.
    """
    try:
        return sum(f.stat().st_size for f in path.glob("*.safetensors"))
    except OSError:
        return 0


class ModelAdmissionError(RuntimeError):
    """A model can't be admitted under the configured memory ceiling. Raised *before* the load so
    a too-big model fails with a clear message (surfaced as a chat error / a 502 on /models load)
    instead of letting MLX OOM-crash the whole backend."""


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


def _render_prompt(
    templater,
    messages: list[dict],
    tools: list[dict] | None,
    template_kwargs: dict | None = None,
) -> str:
    """Render the chat prompt, normalising tool_calls first and falling back ONLY when the
    tokenizer genuinely rejects the ``tools`` kwarg.

    A TypeError from *inside* the template (a message-shape mismatch) must surface — retrying
    without tools would just fail the same way and mask the real cause. The previous blanket
    ``except TypeError`` swallowed exactly that, hiding the Qwen3.x tool_calls render bug.

    ``template_kwargs`` are forwarded into the chat template's jinja context (e.g. Qwen3.x's
    ``enable_thinking=False``, which swaps the generation prompt's open ``<think>`` for an empty
    block so the model answers directly). Templates that don't know a variable ignore it.
    """
    messages = _messages_for_template(messages)
    extra = template_kwargs or {}
    try:
        return templater(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **extra
        )
    except TypeError as exc:
        if "tools" not in str(exc):  # not the "template doesn't accept tools" case — surface it
            raise
        return templater(messages, add_generation_prompt=True, tokenize=False, **extra)


def _sampler_kwargs(
    temperature: float | None, top_p: float | None, top_k: int | None
) -> dict:
    """Build mlx-lm's ``sampler`` kwarg from per-model overrides, or {} when none are set (so
    the library keeps its own greedy/default sampling). Isolated so the mlx-lm import stays
    lazy and a sampler-API change touches only here."""
    if temperature is None and top_p is None and top_k is None:
        return {}
    from mlx_lm.sample_utils import make_sampler

    return {
        "sampler": make_sampler(
            temp=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 0.0,
            top_k=int(top_k) if top_k is not None else 0,
        )
    }


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
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        chat_template_kwargs: dict | None = None,
        **_ignored,
    ) -> Iterator[str]:
        # Imported lazily: MLX is a heavy, Apple-Silicon-only dependency.
        from mlx_lm import stream_generate

        # Passing tools lets the chat template render them in the model's expected
        # tool-calling format. _render_prompt normalises tool_calls and handles the
        # tools-kwarg fallback without masking real template errors.
        prompt = _render_prompt(
            self._tokenizer.apply_chat_template, messages, tools, chat_template_kwargs
        )
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            **_sampler_kwargs(temperature, top_p, top_k),
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
        chat_template_kwargs: dict | None = None,
        **_ignored,
    ) -> Iterator[str]:
        from mlx_vlm import stream_generate

        templater = getattr(self._processor, "apply_chat_template", None) or getattr(
            self._processor, "tokenizer"
        ).apply_chat_template
        prompt = _render_prompt(templater, messages, tools, chat_template_kwargs)
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
    try:
        model, processor = load(str(path))
    except Exception as exc:
        # mlx-vlm loads weights strictly: a checkpoint whose architecture doesn't match
        # mlx-vlm's class for its model_type raises a ValueError that lists EVERY mismatched
        # weight — hundreds of names (e.g. Lance-3B-Video, a custom "omni" understand+generate
        # variant declaring model_type qwen2_5_vl but carrying extra gen heads/VAE bridges the
        # stock class lacks). Collapse it to a one-line, actionable error so the gateway/GUI
        # shows a usable message instead of a multi-KB key dump; the full detail stays in the
        # backend log via the caller's log.exception.
        head = next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), "") or type(
            exc
        ).__name__
        raise RuntimeError(
            f"this vision-language model failed to load — its checkpoint doesn't match "
            f"mlx-vlm's architecture ({head}). It's likely a custom or unsupported variant; "
            f"pick a standard chat model instead."
        ) from exc
    return VlmChatEngine(model, processor)


def _default_loader(path: Path, forced_kind: str | None = None) -> object:
    # Dispatch by model family: a VL/omni checkpoint loads through mlx-vlm, everything else
    # through mlx-lm. ``forced_kind`` (a per-model type override) wins over auto-detection — it is
    # how a checkpoint we'd misclassify as VLM, and then crash the mlx-vlm loader on, is told to
    # load as a plain LLM and just work. When unset, classify_kind re-reads config.json from path.
    from .mlx_discovery import classify_kind

    kind = forced_kind or classify_kind(path)
    return _load_vlm(path) if kind == "vlm" else _load_llm(path)


class MlxEnginePool:
    """LRU pool of loaded engines. ``loader`` is injectable for tests."""

    def __init__(
        self,
        *,
        max_loaded: int | None = 1,
        loader: Callable[..., object] | None = None,
        pinned: set[str] | None = None,
        mem_ceiling_bytes: int | None = None,
    ):
        # None / <=0 means "no count cap — the memory ceiling alone gates residency", which is
        # what lets several models stay resident when they fit (fusion prefetch, fast switching).
        self._max = max_loaded if (max_loaded and max_loaded > 0) else None
        self._loader = loader or _default_loader
        # Insertion order == LRU order; move_to_end marks most-recently-used.
        self._loaded: OrderedDict[str, object] = OrderedDict()
        # model_id -> bytes it's holding (measured active-memory delta, or disk estimate as a
        # fallback). Drives byte-level admission against the ceiling, alongside the count cap.
        self._footprint: dict[str, int] = {}
        # model_id -> its dedicated single-thread executor. MLX (mlx-vlm especially) binds a GPU
        # stream to the thread a model was LOADED on; generating from any other thread raises
        # "There is no Stream(gpu, N) in current thread". So every engine gets one thread for its
        # whole life, and both the load and all generation run there (thread affinity).
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._pinned: set[str] = set(pinned or set())
        # None / <=0 disables byte-admission entirely — the pool then behaves exactly as the
        # count-only LRU it was before, so the guardrail is strictly opt-in via mem_ceiling_gb.
        self._ceiling = mem_ceiling_bytes if (mem_ceiling_bytes and mem_ceiling_bytes > 0) else None
        self._lock = asyncio.Lock()

    def _effective_max(self) -> int | None:
        """Count cap actually enforced: the configured one, or — when neither a cap nor a memory
        ceiling exists — a failsafe of 1 so an unbounded pool can't OOM a machine whose RAM we
        couldn't detect. None means "no count cap" (ceiling-gated)."""
        if self._max is not None:
            return self._max
        return None if self._ceiling is not None else 1

    async def acquire(
        self, model_id: str, path: Path, forced_kind: str | None = None
    ) -> object:
        """Return the loaded engine for ``model_id``, loading (and evicting) as needed.
        ``forced_kind`` overrides the loader's auto-detection (per-model type override).

        Raises ``ModelAdmissionError`` *before* loading if a memory ceiling is set and the model
        can't be made to fit — fail loud rather than OOM-crash."""
        # Lock-free fast path: pool state only mutates from the event loop and this path has no
        # awaits, so it's atomic w.r.t. other coroutines. Without it, a resident model's acquire
        # would queue behind whatever load currently holds the lock — i.e. generation on model A
        # would stall for the full duration of model B's background prefetch.
        engine = self._loaded.get(model_id)
        if engine is not None:
            self._loaded.move_to_end(model_id)
            return engine
        async with self._lock:
            if model_id in self._loaded:  # loaded while we waited (e.g. by its own prefetch)
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            incoming = _estimate_model_bytes(path)
            self._make_room(exclude=model_id, incoming=incoming)
            # Load outside would race the lock; loading under the lock serialises
            # model switches, which is what we want (one heavy load at a time).
            log.info("loading model into pool: %s (~%.1fGB est)", model_id, incoming / 1e9)
            before = _active_memory_bytes()
            # Load on the model's OWN thread — generation must later run on this same thread
            # (mlx-vlm stream affinity), so the executor is created first and the load is its
            # first job. Torn down when the model leaves the pool.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mlx-{model_id}")
            try:
                engine = await asyncio.get_running_loop().run_in_executor(
                    executor, self._loader, path, forced_kind
                )
            except BaseException:
                executor.shutdown(wait=False)
                raise
            self._executors[model_id] = executor
            self._loaded[model_id] = engine
            self._loaded.move_to_end(model_id)
            # Refine the disk estimate with the real footprint when MLX exposes it. The lock is
            # held and the cache was just cleared on eviction, so the active-memory delta is this
            # model's own contribution; fall back to the estimate (tests / non-MLX / no delta).
            after = _active_memory_bytes()
            measured = (after - before) if (before is not None and after is not None and after > before) else 0
            self._footprint[model_id] = measured or incoming
            log.info(
                "model loaded: %s (%.1fGB, pool now: %s = %.1fGB%s)",
                model_id, self._footprint[model_id] / 1e9, self.loaded_ids(),
                self._loaded_bytes() / 1e9,
                f"/{self._ceiling / 1e9:.0f}GB ceiling" if self._ceiling else "",
            )
            return engine

    def _loaded_bytes(self, exclude: str | None = None) -> int:
        return sum(self._footprint.get(k, 0) for k in self._loaded if k != exclude)

    def _drop(self, model_id: str, reason: str) -> None:
        self._loaded.pop(model_id, None)
        self._footprint.pop(model_id, None)
        # wait=False: an in-flight generation on this thread finishes on its own (the worker
        # holds the engine reference); the executor just stops accepting new work.
        executor = self._executors.pop(model_id, None)
        if executor is not None:
            executor.shutdown(wait=False)
        log.info("evicted model from pool: %s (%s)", model_id, reason)

    def _make_room(self, exclude: str, incoming: int) -> None:
        """Evict LRU non-pinned engines until both the count cap (``max_loaded``) and, when a
        ceiling is set, the memory budget (held + ``incoming`` <= ceiling) are satisfied. Dropping
        the last reference is necessary but not sufficient to free MLX's unified memory —
        ``_release_mlx_memory`` (once, after any eviction) clears the Metal buffer cache too.

        Raises ``ModelAdmissionError`` if, after evicting everything evictable, the incoming model
        still overflows the ceiling (it's larger than the budget, or pinned models leave no room)."""
        evicted = False

        def next_victim() -> str | None:
            return next((k for k in self._loaded if k != exclude and k not in self._pinned), None)

        # Count cap: make room for one more (ceiling-independent — preserves prior behaviour).
        # No cap (None) → residency is governed by the memory ceiling alone.
        cap = self._effective_max()
        while cap is not None and len(self._loaded) >= cap:
            victim = next_victim()
            if victim is None:
                break  # everything left is pinned — exceed the count budget rather than evict it
            self._drop(victim, f"making room for {exclude}")
            evicted = True

        # Memory ceiling: keep evicting until the incoming estimate fits under the budget.
        if self._ceiling is not None:
            while self._loaded_bytes(exclude) + incoming > self._ceiling:
                victim = next_victim()
                if victim is None:
                    break
                self._drop(victim, f"freeing memory for {exclude}")
                evicted = True

        if evicted:
            _release_mlx_memory()

        if self._ceiling is not None and self._loaded_bytes(exclude) + incoming > self._ceiling:
            held = self._loaded_bytes(exclude)
            raise ModelAdmissionError(
                f"{exclude} needs ~{incoming / 1e9:.1f}GB but only "
                f"{max(0.0, (self._ceiling - held) / 1e9):.1f}GB is free under the "
                f"{self._ceiling / 1e9:.0f}GB memory limit"
                + (f" ({held / 1e9:.1f}GB held by pinned models)" if held else "")
                + ". Free memory (unload other models / close apps) or pick a smaller model."
            )

    async def load(self, model_id: str, path: Path, forced_kind: str | None = None) -> None:
        await self.acquire(model_id, path, forced_kind)

    async def unload(self, model_id: str) -> bool:
        async with self._lock:
            removed = self._loaded.pop(model_id, None) is not None
            if removed:
                self._footprint.pop(model_id, None)
                executor = self._executors.pop(model_id, None)
                if executor is not None:
                    executor.shutdown(wait=False)
                log.info("unloading model from pool: %s (pool now: %s)", model_id, self.loaded_ids())
                _release_mlx_memory()
            else:
                log.info("unload requested for model not in pool: %s", model_id)
            return removed

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def loaded_ids(self) -> list[str]:
        return list(self._loaded.keys())

    def executor_for(self, model_id: str) -> ThreadPoolExecutor | None:
        """The dedicated thread a loaded model must generate on (see ``_executors``); None when
        the model isn't resident. Callers submit generation here rather than to a shared pool —
        running it on any other thread breaks mlx-vlm's per-thread GPU stream."""
        return self._executors.get(model_id)

    def headroom_bytes(self) -> int | None:
        """Bytes still admittable WITHOUT evicting anything, or None for "unbounded". Lets a
        scheduler (fusion prefetch) ask "would another model fit alongside what's resident?" —
        prefetching must never evict, because the victim may be mid-generation: its memory stays
        held by the running worker while the pool's books say it's free. A full count cap means
        the next load *will* evict, so headroom is 0 regardless of free bytes."""
        cap = self._effective_max()
        if cap is not None and len(self._loaded) >= cap:
            return 0
        if self._ceiling is None:
            return None
        return max(0, self._ceiling - self._loaded_bytes())

    def set_mem_ceiling_bytes(self, ceiling: int | None) -> None:
        """Live-update the admission ceiling (GUI Settings edit; the next acquire enforces it, no
        restart). None / <=0 disables byte-admission. Already-loaded models aren't evicted to fit a
        newly-lowered ceiling — it bites on the next load, which is when OOM risk actually arrives."""
        self._ceiling = ceiling if (ceiling and ceiling > 0) else None

    def pin(self, model_id: str) -> None:
        self._pinned.add(model_id)

    def unpin(self, model_id: str) -> None:
        self._pinned.discard(model_id)
