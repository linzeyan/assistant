"""Concurrent generation lane for the native MLX backend (N104).

Lets ONE loaded model serve several requests at once: mlx-lm's ``BatchGenerator``
steps every active request through a single batched decode loop, so N Claude Code
subagents fanning out over ``/v1/messages`` decode together instead of queueing
turn-by-turn behind the engine's serial path. Targets the mlx-lm >=0.31 API
(``next()`` returns (prompt_responses, generation_responses); per-sequence samplers
on ``insert``; finished responses carry their extracted ``prompt_cache`` +
``all_tokens``) — the batchability gate treats an older mlx-lm as "can't batch" and
falls back to serial rather than crashing mid-burst.

Deliberate scope:
- Only the Anthropic compat route and the subagent runner opt in
  (``concurrent=True``). GUI / Telegram main turns keep the serial path and its
  per-conversation cache singleton — interactive one-at-a-time flows gain nothing
  from batching and shouldn't inherit its constraints.
- Sampling params ride per-request (``samplers=[...]`` at insert), so requests with
  different temperatures share one batch — no drain barrier.
- Only engines whose cache layout is batchable join (KVCache / RotatingKVCache —
  mirroring upstream's server gate); everything else, including VLM engines, falls
  back to serial.
- Finished requests bank their KV cache in a small per-model LRU keyed by token
  ids — the batched counterpart of ``MlxEngine._prefill_plan`` — so a
  conversation's next turn prefills only its new tail.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from .mlx_engine import (
    PromptCacheStore,
    _active_memory_bytes,
    _peak_memory_bytes,
    _reset_peak_memory,
)

log = logging.getLogger(__name__)


def _default_generator_factory(engine: object, max_concurrent: int) -> object:
    """A ``BatchGenerator`` for this engine. ``prefill_batch_size=1`` so a free slot
    admits a waiting request immediately — admission latency matters more here than
    prefill batching, because subagents arrive staggered and each extra wait is a whole
    prompt's prefill. Stop tokens are the tokenizer's EOS set as single-token stop
    sequences (the >=0.31 state-machine form)."""
    from mlx_lm.generate import BatchGenerator

    return BatchGenerator(
        engine._model,
        stop_tokens=[[t] for t in engine._tokenizer.eos_token_ids],
        completion_batch_size=max_concurrent,
        prefill_batch_size=1,
    )


def _sampler_for(sampling: tuple) -> object | None:
    """Per-request sampler from (temperature, top_p, top_k); None = generator's greedy
    fallback, matching the serial path's default when no params are set."""
    temperature, top_p, top_k = sampling
    if temperature is None and top_p is None and top_k is None:
        return None
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(
        temp=float(temperature) if temperature is not None else 0.0,
        top_p=float(top_p) if top_p is not None else 0.0,
        top_k=int(top_k) if top_k is not None else 0,
    )


@dataclass
class BatchRequest:
    """One request's bridge between the event loop and the burst thread. The queue speaks the
    serial worker's exact protocol — ("token", str) / ("error", exc) / ("end", None) — so the
    consumer state machine in ``MlxModelService._stream`` needs no batch-specific code."""

    loop: object  # asyncio event loop (call_soon_threadsafe target)
    queue: object  # asyncio.Queue
    stop: threading.Event
    usage: dict
    messages: list
    tools: list | None
    sampling: tuple  # (temperature, top_p, top_k) — becomes this request's own sampler
    max_tokens: int
    template_kwargs: dict | None

    def post(self, item: tuple) -> None:
        self.loop.call_soon_threadsafe(self.queue.put_nowait, item)


@dataclass
class _Active:
    """Burst-thread state for one admitted request."""

    req: BatchRequest
    detok: object  # per-request streaming detokenizer (stateful — never shared)
    prompt_tokens: int  # full prompt length (usage + logging)
    prefill: int  # tokens actually prefilled (cache-miss cost, for the log line)
    count: int = 0  # generated tokens, excluding the stop token
    t0: float = field(default_factory=time.monotonic)


class BatchLane:
    """Per-engine burst runner. ``submit`` is event-loop-side and thread-safe against the
    burst thread's idle-exit; ``run`` occupies the model's pool thread only while there is
    work, then exits so serial jobs and the pool's retirement clear can use the thread."""

    def __init__(
        self,
        engine: object,
        max_concurrent: int,
        *,
        generator_factory=None,
        sampler_factory=None,
        trimmer=None,
    ):
        self._engine = engine
        self._max = max(1, max_concurrent)
        self._factory = generator_factory or _default_generator_factory
        self._sampler = sampler_factory or _sampler_for
        self._store = PromptCacheStore(trimmer=trimmer)
        self._pending: deque[BatchRequest] = deque()
        self._running = False
        self._lock = threading.Lock()

    # --- event-loop side ---

    def start(
        self,
        loop,
        queue,
        stop: threading.Event,
        usage: dict,
        executor,
        messages: list,
        tools: list | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        chat_template_kwargs: dict | None = None,
        **_ignored,
    ) -> None:
        """Enqueue one request and make sure a burst is running on the model's thread.
        Mirrors ``MlxEngine.stream_text``'s signature so ``_stream`` forwards params as-is."""
        req = BatchRequest(
            loop=loop, queue=queue, stop=stop, usage=usage, messages=messages, tools=tools,
            sampling=(temperature, top_p, top_k), max_tokens=max_tokens,
            template_kwargs=chat_template_kwargs,
        )
        if self.submit(req):
            fut = loop.run_in_executor(executor, self.run)
            fut.add_done_callback(_surface_burst_crash)

    def submit(self, req: BatchRequest) -> bool:
        """True → the caller must start ``run`` (no burst is live). The lock closes the race
        against a burst deciding to idle-exit while this request is being enqueued."""
        with self._lock:
            self._pending.append(req)
            if self._running:
                return False
            self._running = True
            return True

    # --- burst-thread side ---

    def _pop(self) -> BatchRequest | None:
        with self._lock:
            return self._pending.popleft() if self._pending else None

    def _idle_exit(self) -> bool:
        """Atomically flip to not-running iff no request slipped in — the counterpart of
        ``submit``'s check, so a request is never left queued with no burst to serve it."""
        with self._lock:
            if self._pending:
                return False
            self._running = False
            return True

    def run(self) -> None:
        eng = self._engine
        # Same working-memory measurement as the serial path, at burst granularity: the pool
        # budgets admissions against a model's measured generation peak (N86).
        resident_before = _active_memory_bytes()
        _reset_peak_memory()
        gen = None
        active: dict[int, _Active] = {}
        try:
            while True:
                # Admit between decode steps: new requests join the running batch
                # (continuous batching). Sampling rides per-request (insert's ``samplers``),
                # so mixed params share one batch — no drain barrier.
                while len(active) < self._max:
                    req = self._pop()
                    if req is None:
                        break
                    if req.stop.is_set():  # cancelled before it ever started
                        req.post(("end", None))
                        continue
                    if gen is None:
                        gen = self._factory(eng, self._max)
                    try:
                        uid, state = self._insert(gen, req, len(active))
                    except Exception as exc:  # render/encode failure is this request's alone
                        req.post(("error", exc))
                        req.post(("end", None))
                        continue
                    active[uid] = state

                if not active:
                    if gen is not None:
                        gen.close()
                        gen = None
                    if self._idle_exit():
                        return
                    continue

                # Prompt-progress responses are dropped: SSE keepalive covers the silence,
                # and per-request prefill cost is logged at finish.
                _, responses = gen.next()
                for r in responses:
                    state = active.get(r.uid)
                    if state is None:
                        continue
                    if r.finish_reason != "stop":
                        state.count += 1
                        state.detok.add_token(r.token)
                        seg = state.detok.last_segment
                        if seg:
                            state.req.post(("token", seg))
                    if r.finish_reason is not None:
                        self._finish(state, r)
                        del active[r.uid]

                cancelled = [uid for uid, st in active.items() if st.req.stop.is_set()]
                if cancelled:
                    # Clean mid-batch removal — the serial path can only cooperatively break
                    # between tokens (N81); here the slot frees this very step.
                    gen.remove(cancelled)
                    for uid in cancelled:
                        active.pop(uid).req.post(("end", None))
        except Exception as exc:
            # A step failure poisons the whole batch (shared forward pass) — fail every rider
            # loudly, and drain pending too: they were promised a running burst.
            log.exception("batch lane burst failed")
            casualties = [st.req for st in active.values()]
            while (req := self._pop()) is not None:
                casualties.append(req)
            for req in casualties:
                req.post(("error", exc))
                req.post(("end", None))
            with self._lock:
                self._running = False
        finally:
            if gen is not None:
                gen.close()
            peak = _peak_memory_bytes()
            if peak is not None and resident_before is not None:
                eng.working_memory_bytes = max(
                    getattr(eng, "working_memory_bytes", 0), peak - resident_before
                )

    def _insert(self, gen, req: BatchRequest, n_active: int) -> tuple[int, _Active]:
        eng = self._engine
        full_ids = eng.encode_prompt(req.messages, req.tools, req.template_kwargs)
        req.usage["input_tokens"] = len(full_ids)
        suffix, cache = self._store.take(full_ids)
        prefix = full_ids[: len(full_ids) - len(suffix)]
        # ``all_tokens`` tells the generator which tokens the passed cache already holds, so
        # the finished Response's ``all_tokens`` is the full sequence — exactly the key the
        # cache bank needs for the next turn's prefix match.
        (uid,) = gen.insert(
            [suffix],
            max_tokens=[req.max_tokens],
            caches=[cache] if cache is not None else None,
            all_tokens=[prefix] if cache is not None else None,
            samplers=[self._sampler(req.sampling)],
        )
        log.info(
            "batch admit: prompt=%d (cached=%d, prefill=%d) batch=%d",
            len(full_ids), len(prefix), len(suffix), n_active + 1,
        )
        return uid, _Active(
            req=req, detok=eng._tokenizer.detokenizer,
            prompt_tokens=len(full_ids), prefill=len(suffix),
        )

    def _finish(self, state: _Active, r) -> None:
        state.detok.finalize()
        tail = state.detok.last_segment
        if tail:
            state.req.post(("token", tail))
        state.req.usage["output_tokens"] = state.count
        # The finished response carries its extracted cache and the token ids that cache
        # represents (prompt + generated, stop included) — bank them for the next turn.
        if r.prompt_cache is not None and r.all_tokens is not None:
            self._store.put(list(r.all_tokens), r.prompt_cache)
        log.info(
            "batch generation: prompt=%d (prefill=%d) decode=%d tok in %.2fs (%s)",
            state.prompt_tokens, state.prefill, state.count,
            time.monotonic() - state.t0, r.finish_reason,
        )
        state.req.post(("end", None))


def _surface_burst_crash(fut) -> None:
    """``run`` handles its own errors; this catches the truly unexpected (executor torn down,
    error inside the except path) so a dead burst never disappears silently."""
    try:
        exc = fut.exception()
    except Exception:
        return
    if exc is not None:
        log.error("batch lane worker died: %r", exc)


def lane_for(engine: object, max_concurrent: int) -> BatchLane | None:
    """The engine's lane, created on first use; None when this engine can't batch (VLM,
    non-batchable cache layout, mlx-lm too old, fakes) — callers fall back to serial."""
    lane = getattr(engine, "_batch_lane", None)
    if lane is not None:
        return lane
    if max_concurrent <= 0 or getattr(engine, "_batch_unsupported", False):
        return None
    if not _batchable(engine):
        # Remember the verdict: probing builds a throwaway prompt cache, no need per turn.
        try:
            engine._batch_unsupported = True
        except AttributeError:
            pass
        return None
    lane = BatchLane(engine, max_concurrent)
    engine._batch_lane = lane
    return lane


def _batchable(engine: object) -> bool:
    """Mirror upstream's server cache gate: only standard KV layouts batch. Also require the
    >=0.31 BatchGenerator shape (``insert_segments``) — an older mlx-lm has an incompatible
    API and must fall back to serial, not crash mid-burst. Our stack never sends upstream's
    other exclusions (draft models, logit_bias, repetition_penalty)."""
    if not (hasattr(engine, "encode_prompt") and hasattr(engine, "_model")):
        return False  # VLM engine (no batch support in mlx-vlm) or a test fake
    try:
        from mlx_lm.generate import BatchGenerator
        from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
    except ImportError:
        return False
    if not hasattr(BatchGenerator, "insert_segments"):  # pre-0.31 API
        return False
    try:
        return all(type(c) in (KVCache, RotatingKVCache) for c in make_prompt_cache(engine._model))
    except Exception:
        return False
