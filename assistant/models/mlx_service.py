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
import logging
import shutil
import threading
from dataclasses import replace
from collections.abc import AsyncIterator
from pathlib import Path

from assistant.agent.fusion import FUSION_MODEL_ID

from assistant.model_traits import CHATTABLE_KINDS

from .mlx_discovery import DiscoveredModel, discover_models, is_mlx_loadable
from .mlx_engine import MlxEnginePool, _estimate_model_bytes, _total_ram_bytes
from .service import ModelService
from .status import BackendState, BackendStatus
from .tool_parsing import (
    HARMONY_CHANNEL,
    TOOL_MARKERS,
    earliest_marker,
    harmony_fields,
    normalize_arguments,
    parse_tool_calls,
)
from .types import ModelInfo

log = logging.getLogger(__name__)

# Fraction of physical RAM the pool may commit to model weights before refusing a load. The slack
# leaves room for the OS, the KV cache and activations (which the on-disk estimate omits).
_RAM_HEADROOM = 0.9

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
        max_concurrent: int = 8,
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
        # Memory ceiling: an explicit mem_ceiling_gb wins; otherwise default to the machine's
        # physical RAM minus headroom, so a model too big for this system fails loud with a clear
        # message BEFORE loading rather than OOM-crashing the backend ("check the resource fits
        # first"). Estimates come from on-disk weights, so a 96GB model on 128GB RAM still admits.
        if mem_ceiling_gb:
            mem_ceiling_bytes: int | None = int(mem_ceiling_gb * 1e9)
        else:
            _ram = _total_ram_bytes()
            mem_ceiling_bytes = int(_ram * _RAM_HEADROOM) if _ram else None
        self._pool = pool or MlxEnginePool(max_loaded=max_loaded, mem_ceiling_bytes=mem_ceiling_bytes)
        # Cap for the concurrent lane (mlx_batch): how many /v1/messages requests may decode
        # together on one model. Each concurrent stream holds its own KV cache in unified
        # memory, so this is a memory knob as much as a throughput one; 0 disables the lane.
        self._max_concurrent = max_concurrent
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

    def set_mem_ceiling(self, gb: float | None) -> None:
        """Live-update the pool's memory-admission ceiling (GUI Settings). GB→bytes; 0/None
        disables. The next model load enforces it — no restart."""
        self._pool.set_mem_ceiling_bytes(int(gb * 1e9) if gb else None)

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
        # Honour a per-model type override: force this model's effective kind (e.g. a checkpoint we
        # auto-detect as VLM, and crash mlx-vlm on, told to load as a plain LLM). One resolution
        # point so the chat-gate, the loader dispatch and list_models all agree.
        if self._per_model is not None:
            override = self._per_model.kind_override(model_id)
            if override and override != entry.kind:
                entry = replace(entry, kind=override)
        return entry

    async def _draft_path_for(self, model_id: str) -> Path | None:
        """Resolve this model's paired speculative drafter (per-model "draft" setting) to the
        drafter checkpoint's path, or None when no drafter is configured. Fail-loud on a bad
        pairing: an unknown drafter id or a non-draft checkpoint raises here, at load time,
        instead of surfacing as an opaque weight-shape error inside mlx-vlm."""
        if self._per_model is None:
            return None
        draft_id = self._per_model.draft_model(model_id)
        if not draft_id:
            return None
        entry = await self._entry_for(draft_id)  # ValueError("unknown model: …") if missing
        if entry.kind != "draft":
            raise ValueError(
                f"'{draft_id}' is a {entry.kind} model, not a speculative drafter — "
                f"pick an MTP/DFlash/EAGLE checkpoint (kind 'draft') or clear the setting."
            )
        return entry.path

    async def _acquire(self, model_id: str, entry: DiscoveredModel):
        """Pool acquire with the model's drafter (if any) resolved — the single path every
        engine-needing call takes, so chat, token counting and explicit loads all get the
        same engine variant for a given configuration."""
        draft_path = await self._draft_path_for(model_id)
        return await self._pool.acquire(
            model_id, entry.path, forced_kind=entry.kind, draft_path=draft_path
        )

    # Kinds usable as a chat model — the shared definition (model_traits.CHATTABLE_KINDS), so the
    # load gate, the GUI picker and the Telegram picker can never drift apart.
    _CHATTABLE_KINDS = CHATTABLE_KINDS

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

        def _eff_kind(m: DiscoveredModel) -> str:
            ov = self._per_model.kind_override(m.id) if self._per_model is not None else None
            return ov or m.kind

        # Only checkpoints an MLX engine can actually load are offered. The catalog itself keeps
        # every discovered dir (so delete-by-id still reaches a phantom entry); this filter is
        # about what the pickers are allowed to show — selecting an unloadable model is the
        # crash N27/N32 documented.
        listable = [m for m in self._catalog.values() if is_mlx_loadable(m)]
        skipped = [m.id for m in self._catalog.values() if not is_mlx_loadable(m)]
        if skipped:
            # Loud, not silent: a model vanishing from the list must be explainable without
            # reading this code (these dirs still occupy disk and are still deletable by id).
            log.info(
                "skipping %d non-MLX-loadable model dir(s): %s",
                len(skipped), ", ".join(sorted(skipped)),
            )
        models = [
            ModelInfo(
                id=m.id, type=_eff_kind(m), loaded=m.id in loaded,
                source=m.source, size_bytes=m.size_bytes,
            )
            for m in listable
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
        await self._acquire(model_id, entry)

    async def unload(self, model_id: str) -> None:
        await self._pool.unload(model_id)

    # --- scheduling introspection (fusion's size-aware load/unload planner) ---
    # FusionEngine discovers these via getattr, so a service without them (omlx) simply runs
    # the panel sequentially with no prefetch/unload planning — same events, fewer smarts.

    def loaded_model_ids(self) -> list[str]:
        return self._pool.loaded_ids()

    def headroom_bytes(self) -> int | None:
        return self._pool.headroom_bytes()

    async def estimate_bytes(self, model_id: str) -> int:
        """What keeping ``model_id`` resident costs, 0 when unknown — weights plus the working
        memory generation needs, i.e. the exact number the pool's admission gate checks, so the
        scheduler and the gate agree on what "fits". Returning bare weights here (as this did
        before the gate started budgeting working memory) let the fusion scheduler prefetch into
        room that wasn't there."""
        try:
            entry = await self._entry_for(model_id)
        except ValueError:
            return 0
        return self._pool.admission_bytes(model_id, _estimate_model_bytes(entry.path))

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

    async def count_tokens(
        self, messages: list[dict], model: str, tools: list[dict] | None = None
    ) -> int | None:
        # Render + encode via the model's own tokenizer/template so the count matches the prompt the
        # model would actually see (including per-model chat_template_kwargs). Reuses the pool: on an
        # active Claude Code session the engine is already loaded, so this is just a tokenizer pass.
        # Fusion is a virtual model with no single engine/tokenizer — nothing to count against.
        if not self.available() or model == FUSION_MODEL_ID:
            return None
        entry = await self._entry_for(model)
        self._require_chat_model(entry)
        engine = await self._acquire(model, entry)
        counter = getattr(engine, "count_tokens", None)
        if counter is None:  # an engine variant that can't count → unknown, never a 500
            return None
        tpl = self._per_model.chat_template_kwargs(model) if self._per_model is not None else None
        return await asyncio.to_thread(counter, messages, tools, tpl)

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
        # Routing flag, not a generation param: /v1/messages sets it so batchable engines can
        # decode several requests at once (Claude Code subagent fan-out). Popped here so it
        # never reaches an engine's stream_text.
        concurrent = bool(params.pop("concurrent", False))
        # The model's saved overrides win over the caller's defaults (e.g. a per-model
        # max_tokens overrides the loop's global cap; temperature/top_p/top_k are added).
        if self._per_model is not None:
            params = {**params, **self._per_model.generation(model)}
            # Template kwargs merge per-key (not replace): a caller's kwargs (e.g. fusion's
            # enable_thinking=False) survive unless the user saved that same key for this model
            # — same stored-wins semantics as the sampler params above.
            stored_tpl = self._per_model.chat_template_kwargs(model)
            if stored_tpl:
                params["chat_template_kwargs"] = {
                    **(params.get("chat_template_kwargs") or {}),
                    **stored_tpl,
                }
        return self._stream(messages, model, tools, known, concurrent=concurrent, **params)

    async def _stream(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None,
        known_names: set[str],
        concurrent: bool = False,
        **params,
    ) -> AsyncIterator[dict]:
        entry = await self._entry_for(model)
        self._require_chat_model(entry)
        engine = await self._acquire(model, entry)
        # Generation MUST run on the thread the model was loaded on: mlx-vlm binds a GPU
        # stream to the load thread and raises "There is no Stream(gpu, N) in current thread"
        # from any other (the bug that silently knocked VLM panel models out of fusion). No
        # await between acquire and here, so the executor can't have been torn down.
        executor = self._pool.executor_for(model)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        # Cooperative cancellation: the blocking engine generator runs on an executor
        # thread that nothing can interrupt, so a client disconnect used to leave it
        # decoding to max_tokens — burning the model's single engine slot for minutes
        # while every later request silently queued behind the zombie (N81). The worker
        # checks this flag each token; breaking closes the engine generator, whose
        # GeneratorExit path already invalidates the KV cache like any early stop.
        stop = threading.Event()
        # Filled by the engine thread as it generates; read after the stream ends to emit a usage
        # event (input/output token counts) so the Anthropic compat route reports real numbers and
        # Claude Code can track how full the context is. Empty for engines that don't count (VLM).
        usage: dict = {}

        # Concurrent lane (mlx_batch): /v1/messages requests on a batchable engine decode
        # together in one batched step loop on the model's thread. Everything else — GUI /
        # Telegram turns, VLM engines, non-batchable cache layouts, mlx-lm too old — takes
        # the serial worker below. The lane speaks the same queue protocol, so the consumer
        # state machine after this block is shared.
        lane = None
        if concurrent:
            from .mlx_batch import lane_for

            lane = lane_for(engine, self._max_concurrent)
        if lane is not None:
            lane.start(loop, queue, stop, usage, executor, messages, tools=tools, **params)
            # No per-request future: the burst outlives this request, and awaiting it in the
            # finally below would pin a cancelled request until the whole batch drains.
            fut = None
            del engine, lane
        else:
            # Bind the engine as a default arg (the worker's own reference) and then drop
            # both `engine` and `worker` from this async generator's frame. Otherwise the
            # frame keeps the engine alive for the generator's whole lifetime — so a later
            # pool.unload() pops _loaded yet gc can't reclaim the model, and its unified
            # memory is never returned (the "Unload doesn't free memory" bug). The executor
            # holds its own reference only until the worker finishes.
            def worker(eng: object = engine) -> None:
                try:
                    for text in eng.stream_text(
                        messages, tools=tools, usage_out=usage, **params
                    ):
                        if stop.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", text))
                except Exception as exc:  # surfaced to the consumer below
                    loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, ("end", None))

            fut = loop.run_in_executor(executor, worker)  # None → default pool (fake-pool tests)
            del engine, lane, worker
        # Streaming state machine: forward prose token-by-token, but suppress any
        # tool-call markup so it never reaches the user — it's re-emitted as a
        # structured tool_calls event once the turn completes. A response that opens
        # with JSON is buffered whole (it may be a bare-JSON tool call).
        buffer = ""
        emitted = 0
        first_char_seen = False
        json_mode = False
        saw_marker = False
        harmony_mode = False
        # name → JSON-Schema properties, for the argument-normalization middleware below.
        schemas = {
            fn.get("name"): (fn.get("parameters") or {}).get("properties") or {}
            for t in (tools or [])
            if (fn := (t.get("function") or {})).get("name")
        }
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
                        # Harmony (gpt-oss) opens every step with <|channel|> (an atomic
                        # special token, so it can't straddle chunks). Its markup
                        # interleaves reasoning, calls, and prose — buffer the whole
                        # step and emit the sanitized split at the end (N84). Keepalive
                        # (N81) keeps the silent buffering from looking like a stall.
                        harmony_mode = stripped.startswith(HARMONY_CHANNEL)
                if json_mode or saw_marker or harmony_mode or not first_char_seen:
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
            if harmony_mode:
                # Raw channel markup must never surface (cf. N3/N67 for other formats).
                # Reasoning rides the product's <think> display convention (the GUI
                # collapses it, N1); the render side maps it back to the template's
                # thinking field so the wire format stays faithful (N82/N84).
                thinking, final = harmony_fields(buffer)
                display = f"<think>{thinking}</think>" if thinking else ""
                if final:
                    display = f"{display}\n\n{final}" if display else final
                if display:
                    yield {"type": "text", "content": display}
            if calls:
                # Normalize each call's arguments against its tool schema (the shared middleware):
                # local models over-quote scalars (e.g. Qwen3-Coder's `"replace_all": "False"`),
                # which the downstream schema validator would reject. schemas maps name→properties.
                yield {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "name": c.name,
                            "arguments": normalize_arguments(
                                c.arguments, schemas.get(c.name, {})
                            ),
                        }
                        for c in calls
                    ],
                }
            elif not saw_marker and not harmony_mode:
                # No call parsed AND no marker was seen: this is genuine prose — flush it.
                remainder = buffer[emitted:]
                if remainder:
                    yield {"type": "text", "content": remainder}
            # else: a tool marker opened but yielded no valid call (truncated at max_tokens/EOS, or
            # malformed) — drop the raw markup instead of leaking a bare "<tool_call>" to the user
            # (the symptom the user saw). Tool syntax must never surface as text (cf. N3/N67).
            if usage:  # engine reported token counts (empty for VLM / non-counting engines)
                yield {"type": "usage", **usage}
        finally:
            stop.set()  # even if the await below is itself cancelled, the thread exits
            if fut is not None:  # serial worker only; a lane burst cleans up its own riders
                await fut
