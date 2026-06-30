"""Native in-process MLX model backend (plan Part A, option A3).

Runs local inference with ``mlx-lm`` directly — no external omlx server. Model
management mirrors omlx: discover models on disk (``mlx_discovery``) and preload /
unload / switch them via an LRU pool (``MlxEnginePool``). Generation is blocking, so
``stream_chat`` runs it in a worker thread and bridges tokens back to the event loop
through a queue, never blocking it.

Tool calling: mlx-lm emits tool calls as text, so ``stream_chat`` streams prose
normally but watches for tool-call markers; once one appears it stops emitting text
and, at end of turn, parses the buffer (``tool_parsing``) into the same structured
``tool_calls`` event the omlx backend produces. This gives the native backend full
parity — the agent's coding / self-learning loop works here too.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from assistant.agent.fusion import FUSION_MODEL_ID

from .mlx_discovery import DiscoveredModel, discover_models
from .mlx_engine import MlxEnginePool
from .service import ModelService
from .status import BackendState, BackendStatus
from .tool_parsing import TOOL_MARKERS, earliest_marker, parse_tool_calls
from .types import ModelInfo

# Hold back this many trailing chars while streaming so a tool-call marker split
# across token boundaries is detected before its prefix leaks as text.
_HOLD = max(len(m) for m in TOOL_MARKERS) - 1

# Config keys a model's context length hides behind, across architectures. VL/omni
# checkpoints nest the text config, so we recurse into those too.
_CTX_KEYS = ("max_position_embeddings", "n_positions", "max_seq_len", "max_sequence_length", "n_ctx")
_CTX_NESTS = ("text_config", "llm_config", "language_config")


def _extract_ctx(cfg: dict) -> int | None:
    for k in _CTX_KEYS:
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            return v
    for nest in _CTX_NESTS:
        sub = cfg.get(nest)
        if isinstance(sub, dict):
            found = _extract_ctx(sub)
            if found:
                return found
    return None


def _read_context_window(path: Path) -> int | None:
    """Read a model's trained context length from its ``config.json`` (no model load needed).
    Returns None when the file is missing/unreadable or names no recognised window key."""
    try:
        cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _extract_ctx(cfg) if isinstance(cfg, dict) else None


class MlxModelService(ModelService):
    def __init__(
        self,
        *,
        models_dir: Path,
        max_loaded: int = 1,
        mem_ceiling_gb: float | None = None,
        include_hf_cache: bool = False,
        extra_model_dirs: list[Path] | None = None,
        pool: MlxEnginePool | None = None,
        available_override: bool | None = None,
        per_model=None,
        fusion=None,
    ):
        self._models_dir = Path(models_dir)
        self._extra_dirs = [Path(d) for d in (extra_model_dirs or [])]
        self._include_hf = include_hf_cache
        # Memory ceiling is opt-in: None leaves the pool count-only (prior behaviour). GB→bytes.
        mem_ceiling_bytes = int(mem_ceiling_gb * 1e9) if mem_ceiling_gb else None
        self._pool = pool or MlxEnginePool(max_loaded=max_loaded, mem_ceiling_bytes=mem_ceiling_bytes)
        # Optional per-model generation overrides (PerModelStore). Merged into stream params so
        # each model's saved temperature/top_p/top_k/max_tokens apply automatically at chat time.
        self._per_model = per_model
        # Optional Fusion engine (panel+judge), surfaced as the virtual "fusion" model.
        self._fusion = fusion
        # available_override lets tests exercise the pool/discovery logic with a fake
        # loader on machines without mlx-lm installed.
        self._available_override = available_override
        self._catalog: dict[str, DiscoveredModel] = {}
        self._status: BackendStatus | None = None

    def available(self) -> bool:
        if self._available_override is not None:
            return self._available_override
        return importlib.util.find_spec("mlx_lm") is not None

    async def start(self) -> BackendStatus:
        if not self.available():
            self._status = BackendStatus(
                BackendState.UNAVAILABLE,
                'mlx-lm not installed. Run: uv pip install -e ".[mlx]"',
                "in-process",
            )
            return self._status
        await self._refresh_catalog()
        self._status = BackendStatus(
            BackendState.LOCAL,
            f"Native MLX backend (mlx-lm); {len(self._catalog)} models discovered.",
            "in-process",
        )
        return self._status

    async def stop(self) -> None:
        for model_id in self._pool.loaded_ids():
            await self._pool.unload(model_id)

    @property
    def status(self) -> BackendStatus | None:
        return self._status

    async def reachable(self) -> bool:
        return self.available()

    def reconfigure(
        self,
        *,
        models_dir: Path,
        extra_model_dirs: list[Path] | None = None,
        include_hf_cache: bool = False,
    ) -> None:
        """Apply discovery settings to the live service — no process restart needed.

        Discovery is just a filesystem scan, so new dirs / the cache toggle can take
        effect immediately; clearing the catalogue forces a re-scan on the next list.
        """
        self._models_dir = Path(models_dir)
        self._extra_dirs = [Path(d) for d in (extra_model_dirs or [])]
        self._include_hf = include_hf_cache
        self._catalog = {}

    async def _refresh_catalog(self) -> None:
        found = await asyncio.to_thread(
            discover_models,
            self._models_dir,
            self._include_hf,
            None,
            self._extra_dirs,
        )
        self._catalog = {m.id: m for m in found}

    async def _entry_for(self, model_id: str) -> DiscoveredModel:
        if model_id not in self._catalog:
            await self._refresh_catalog()  # model may have appeared since last scan
        entry = self._catalog.get(model_id)
        if entry is None:
            raise ValueError(f"unknown model: {model_id}")
        return entry

    # Kinds usable as a chat model: text LLMs (mlx-lm) and vision-language / omni
    # checkpoints (mlx-vlm, text-only chat). Embeddings / cached diffusion text-encoders
    # aren't generative, so they'd crash the loader — refuse them with a clear message.
    _CHATTABLE_KINDS = frozenset({"llm", "vlm"})

    @classmethod
    def _require_chat_model(cls, entry: DiscoveredModel) -> None:
        if entry.kind not in cls._CHATTABLE_KINDS:
            raise ValueError(
                f"'{entry.id}' is a {entry.kind} model — it can't be used as a chat "
                f"model (only text LLMs and vision-language models can)."
            )

    async def list_models(self) -> list[ModelInfo]:
        # Fail soft (mirrors OmlxModelService): no backend → empty list, never raise.
        if not self.available():
            return []
        await self._refresh_catalog()
        loaded = set(self._pool.loaded_ids())
        models = [
            ModelInfo(
                id=m.id, type=m.kind, loaded=m.id in loaded,
                source=m.source, size_bytes=m.size_bytes,
            )
            for m in self._catalog.values()
        ]
        # Surface Fusion as a selectable virtual model (panel+judge) when configured.
        if self._fusion is not None and self._fusion.enabled:
            models.insert(
                0,
                ModelInfo(
                    id=FUSION_MODEL_ID, type="llm", loaded=False,
                    source="virtual", size_bytes=0,
                ),
            )
        return models

    async def delete(self, model_id: str) -> None:
        """Remove a model's files from disk. Only models the user placed in their own
        model directories are deletable here — refuse shared HF-cache entries (other
        tools rely on them) so a delete never nukes something we didn't manage."""
        entry = await self._entry_for(model_id)
        if entry.source != "local":
            raise ValueError(
                f"'{model_id}' lives in the HuggingFace cache, not your model "
                f"directory — delete it with `hf cache` tools, not here."
            )
        if model_id in self._pool.loaded_ids():
            await self._pool.unload(model_id)
        await asyncio.to_thread(shutil.rmtree, entry.path, ignore_errors=True)
        self._catalog.pop(model_id, None)

    async def load(self, model_id: str) -> None:
        if model_id == FUSION_MODEL_ID:
            return  # virtual model: nothing to load (its panel models load on demand)
        entry = await self._entry_for(model_id)
        self._require_chat_model(entry)
        await self._pool.load(model_id, entry.path)

    async def unload(self, model_id: str) -> None:
        await self._pool.unload(model_id)

    async def context_window(self, model: str) -> int | None:
        # Read it from the model's own config.json (off-thread) — no load required, works
        # whether or not the model is currently in the pool. Unknown model → None (fallback).
        if not self.available():
            return None
        try:
            entry = await self._entry_for(model)
        except ValueError:
            return None
        return await asyncio.to_thread(_read_context_window, entry.path)

    def stream_chat(
        self, messages: list[dict], model: str, tools: list[dict] | None = None, **params
    ) -> AsyncIterator[dict]:
        # The virtual "fusion" model runs panel+judge instead of a single engine. It gets no
        # tools (first cut is text-only) and isn't subject to per-model overrides.
        if model == FUSION_MODEL_ID and self._fusion is not None and self._fusion.enabled:
            return self._fusion.answer(
                self, messages, max_tokens=params.get("max_tokens", 1024)
            )
        known = {
            t["function"]["name"]
            for t in (tools or [])
            if isinstance(t.get("function"), dict) and t["function"].get("name")
        }
        # The model's saved overrides win over the caller's defaults (e.g. a per-model
        # max_tokens overrides the loop's global cap; temperature/top_p/top_k are added).
        if self._per_model is not None:
            params = {**params, **self._per_model.get(model)}
        return self._stream(messages, model, tools, known, **params)

    async def _stream(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        known_names: set[str],
        **params,
    ) -> AsyncIterator[dict]:
        entry = await self._entry_for(model)
        self._require_chat_model(entry)
        engine = await self._pool.acquire(model, entry.path)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        # Bind the engine as a default arg (the worker's own reference) and then drop
        # both `engine` and `worker` from this async generator's frame. Otherwise the
        # frame keeps the engine alive for the generator's whole lifetime — so a later
        # pool.unload() pops _loaded yet gc can't reclaim the model, and its unified
        # memory is never returned (the "Unload doesn't free memory" bug). The executor
        # holds its own reference only until the worker finishes.
        def worker(eng: object = engine) -> None:
            try:
                for text in eng.stream_text(messages, tools=tools, **params):
                    loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
            except Exception as exc:  # surfaced to the consumer below
                loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

        fut = loop.run_in_executor(None, worker)
        del engine, worker
        # Streaming state machine: forward prose token-by-token, but suppress any
        # tool-call markup so it never reaches the user — it's re-emitted as a
        # structured tool_calls event once the turn completes. A response that opens
        # with JSON is buffered whole (it may be a bare-JSON tool call).
        buffer = ""
        emitted = 0
        first_char_seen = False
        json_mode = False
        saw_marker = False
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "error":
                    raise payload
                if kind == "end":
                    break

                buffer += payload
                if not first_char_seen:
                    stripped = buffer.lstrip()
                    if stripped:
                        first_char_seen = True
                        json_mode = stripped[0] in "{["
                if json_mode or saw_marker or not first_char_seen:
                    continue  # buffer silently; classify at end

                pos = earliest_marker(buffer, emitted)
                if pos is not None:
                    if pos > emitted:
                        yield {"type": "text", "content": buffer[emitted:pos]}
                    emitted = pos
                    saw_marker = True
                    continue
                # Emit all but a short tail (a marker may straddle token boundaries).
                safe = len(buffer) - _HOLD
                if safe > emitted:
                    yield {"type": "text", "content": buffer[emitted:safe]}
                    emitted = safe

            calls = parse_tool_calls(buffer, known_names=known_names)
            if calls:
                yield {
                    "type": "tool_calls",
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in calls
                    ],
                }
            else:
                remainder = buffer[emitted:]
                if remainder:
                    yield {"type": "text", "content": remainder}
        finally:
            await fut
