"""Batch lane (mlx_batch): concurrent decoding of several requests on one model.

The BatchGenerator itself is mlx-lm's (>=0.31 API: ``next()`` returns
(prompt_responses, generation_responses), per-sequence samplers, finished responses
carry prompt_cache + all_tokens); these tests fake it and verify the lane's own
responsibilities — burst lifecycle, continuous admission, per-request sampling,
mid-batch cancellation, per-conversation cache reuse, and failure propagation.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from assistant.models.mlx_batch import BatchLane, BatchRequest, PromptCacheStore, lane_for

EOS = 999


class FakeDetok:
    """Maps each token to "<n>" so tests can assert exact stream content."""

    def __init__(self):
        self._seg = ""

    def add_token(self, token):
        self._seg += f"<{token}>"

    @property
    def last_segment(self):
        seg, self._seg = self._seg, ""
        return seg

    def finalize(self):
        pass


class FakeTokenizer:
    eos_token_ids = {EOS}

    @property
    def detokenizer(self):
        return FakeDetok()


class FakeEngine:
    def __init__(self):
        self._tokenizer = FakeTokenizer()
        self._model = object()
        self.working_memory_bytes = 0

    def encode_prompt(self, messages, tools=None, chat_template_kwargs=None):
        # Deterministic ids from message text so cache-prefix tests can craft overlaps.
        text = "".join(m.get("content", "") for m in messages)
        return [ord(c) for c in text] or [1]


class FakeBatchGen:
    """Scripted BatchGenerator (>=0.31 contract): each insert consumes the next script (a
    token list); next() yields ([], one response per active uid). EOS finishes with "stop",
    an exhausted script with "length"; finished responses carry prompt_cache + all_tokens."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self._active: dict[int, dict] = {}
        self._uid = 0
        self.inserted: list[tuple[list, object, list | None, object]] = []
        self.removed: list[int] = []
        self.closed = False

    def insert(self, prompts, max_tokens=None, caches=None, all_tokens=None, samplers=None):
        uids = []
        for i, p in enumerate(prompts):
            prefix = list(all_tokens[i]) if all_tokens else []
            self._active[self._uid] = {
                "script": list(self._scripts.pop(0)),
                "tokens": prefix + list(p),  # mirrors upstream: cache contents so far
            }
            self.inserted.append((
                list(p),
                caches[i] if caches else None,
                list(all_tokens[i]) if all_tokens else None,
                samplers[i] if samplers else None,
            ))
            uids.append(self._uid)
            self._uid += 1
        return uids

    def next(self):
        out = []
        for uid in list(self._active):
            st = self._active[uid]
            t = st["script"].pop(0)
            st["tokens"].append(t)
            fin = "stop" if t == EOS else ("length" if not st["script"] else None)
            out.append(SimpleNamespace(
                uid=uid, token=t, finish_reason=fin,
                prompt_cache=f"cache-{uid}" if fin else None,
                all_tokens=list(st["tokens"]) if fin else None,
            ))
            if fin:
                del self._active[uid]
        return [], out

    def remove(self, uids, return_prompt_caches=False):
        self.removed.extend(uids)
        for u in uids:
            self._active.pop(u, None)
        return {}

    def close(self):
        self.closed = True


def _request(loop, content="hi", *, stop=None, temperature=None, max_tokens=64):
    return BatchRequest(
        loop=loop, queue=asyncio.Queue(), stop=stop or threading.Event(), usage={},
        messages=[{"role": "user", "content": content}], tools=None,
        sampling=(temperature, None, None), max_tokens=max_tokens, template_kwargs=None,
    )


async def _collect(req):
    tokens, errors = [], []
    while True:
        kind, payload = await req.queue.get()
        if kind == "end":
            return tokens, errors
        (tokens if kind == "token" else errors).append(payload)


async def test_two_requests_decode_in_one_burst():
    loop = asyncio.get_running_loop()
    gens = []

    def factory(engine, cap):
        gen = FakeBatchGen([[10, 11, EOS], [20, 21, EOS]])
        gens.append(gen)
        return gen

    lane = BatchLane(FakeEngine(), 4, generator_factory=factory)
    r1, r2 = _request(loop, "a"), _request(loop, "b")
    # Enqueue both BEFORE running so the single burst provably serves them together.
    assert lane.submit(r1) is True  # first submit must start a burst
    assert lane.submit(r2) is False  # second rides the running one
    _, (t1, e1), (t2, e2) = await asyncio.gather(
        asyncio.to_thread(lane.run), _collect(r1), _collect(r2)
    )
    assert (t1, e1) == (["<10>", "<11>"], [])
    assert (t2, e2) == (["<20>", "<21>"], [])
    assert len(gens) == 1 and gens[0].closed
    assert r1.usage == {"input_tokens": 1, "output_tokens": 2}
    assert r2.usage == {"input_tokens": 1, "output_tokens": 2}


async def test_mixed_sampling_shares_one_burst_via_per_request_samplers():
    # >=0.31 passes samplers per insert, so different temperatures need no drain barrier:
    # one generator serves both, and each request's sampler reflects its own params.
    loop = asyncio.get_running_loop()
    gens = []

    def factory(engine, cap):
        gen = FakeBatchGen([[10, EOS], [20, EOS]])
        gens.append(gen)
        return gen

    lane = BatchLane(
        FakeEngine(), 4, generator_factory=factory,
        # Injected: the real _sampler_for imports mlx-lm, absent on CI runners.
        sampler_factory=lambda s: None if s == (None, None, None) else ("sampler", s),
    )
    r1 = _request(loop, "a")  # no params → greedy fallback (sampler None)
    r2 = _request(loop, "b", temperature=0.7)
    lane.submit(r1)
    lane.submit(r2)
    _, (t1, _), (t2, _) = await asyncio.gather(
        asyncio.to_thread(lane.run), _collect(r1), _collect(r2)
    )
    assert t1 == ["<10>"] and t2 == ["<20>"]
    assert len(gens) == 1  # one shared burst, NOT a drain-and-restart
    samplers = [s for _, _, _, s in gens[0].inserted]
    assert samplers[0] is None  # default sampling → generator's greedy fallback
    assert samplers[1] == ("sampler", (0.7, None, None))  # its own per-request sampler


async def test_cancellation_removes_from_batch():
    loop = asyncio.get_running_loop()
    gate = threading.Event()

    class GatedGen(FakeBatchGen):
        """Blocks after the first step until the consumer reacted, so the cancel flag is
        provably observed mid-batch rather than racing the script's natural end."""

        def __init__(self, scripts):
            super().__init__(scripts)
            self._steps = 0

        def next(self):
            if self._steps > 0:
                gate.wait(timeout=5)
            self._steps += 1
            return super().next()

    gen_holder = []

    def factory(engine, cap):
        gen = GatedGen([[10] * 10_000])
        gen_holder.append(gen)
        return gen

    lane = BatchLane(FakeEngine(), 4, generator_factory=factory)
    stop = threading.Event()
    req = _request(loop, "a", stop=stop)
    lane.submit(req)

    async def cancel_after_first_token():
        kind, _ = await req.queue.get()
        assert kind == "token"
        stop.set()
        gate.set()
        while True:  # drain to the end marker
            kind, _ = await req.queue.get()
            if kind == "end":
                return

    await asyncio.gather(asyncio.to_thread(lane.run), cancel_after_first_token())
    assert gen_holder[0].removed  # removed mid-batch, not run to completion


async def test_finished_cache_is_reused_for_next_turn():
    loop = asyncio.get_running_loop()
    gens = []

    def factory(engine, cap):
        gen = FakeBatchGen([[10, EOS]] if not gens else [[30, EOS]])
        gens.append(gen)
        return gen

    trims = []
    lane = BatchLane(
        FakeEngine(), 4, generator_factory=factory,
        trimmer=lambda cache, drop: trims.append((cache, drop)) or drop,
    )
    r1 = _request(loop, "abc")  # ids [97, 98, 99]
    lane.submit(r1)
    await asyncio.gather(asyncio.to_thread(lane.run), _collect(r1))
    # Turn 1 banked cache-0 for all_tokens [97, 98, 99, 10, 999] (prompt+generated+stop).
    # Turn 2 resends that exact history plus new text — only the tail should prefill, and
    # the prefix must ride along as all_tokens so the next bank key stays complete.
    r2 = _request(loop, "abc" + chr(10) + chr(EOS) + "Z")
    lane.submit(r2)
    await asyncio.gather(asyncio.to_thread(lane.run), _collect(r2))
    prompt, cache, all_tokens, _sampler = gens[1].inserted[0]
    assert (prompt, cache) == ([ord("Z")], "cache-0")
    assert all_tokens == [97, 98, 99, 10, EOS]  # what the reused cache already holds
    assert trims == []  # exact-prefix hit: nothing to trim
    assert r2.usage["input_tokens"] == 6  # full prompt, regardless of cache reuse


async def test_step_failure_fails_every_rider():
    loop = asyncio.get_running_loop()

    class ExplodingGen(FakeBatchGen):
        def next(self):
            raise RuntimeError("metal fell over")

    lane = BatchLane(
        FakeEngine(), 4, generator_factory=lambda e, c: ExplodingGen([[10], [20]])
    )
    r1, r2 = _request(loop, "a"), _request(loop, "b")
    lane.submit(r1)
    lane.submit(r2)
    _, (t1, e1), (t2, e2) = await asyncio.gather(
        asyncio.to_thread(lane.run), _collect(r1), _collect(r2)
    )
    assert t1 == [] and t2 == []
    assert [str(e) for e in e1] == ["metal fell over"]
    assert [str(e) for e in e2] == ["metal fell over"]
    # The lane must have reset to idle so a later submit starts a fresh burst.
    assert lane.submit(_request(loop, "c")) is True


async def test_lane_exits_idle_and_restarts_for_late_request():
    loop = asyncio.get_running_loop()
    scripts = [[[10, EOS]], [[20, EOS]]]
    lane = BatchLane(
        FakeEngine(), 4, generator_factory=lambda e, c: FakeBatchGen(scripts.pop(0))
    )
    r1 = _request(loop, "a")
    lane.submit(r1)
    await asyncio.gather(asyncio.to_thread(lane.run), _collect(r1))
    r2 = _request(loop, "b")
    assert lane.submit(r2) is True  # burst ended → a late request needs a new one
    _, (t2, _) = await asyncio.gather(asyncio.to_thread(lane.run), _collect(r2))
    assert t2 == ["<20>"]


def _make_model(tmp_path, name: str):
    import json

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"\x00")  # discovery requires real weights
    return d


async def test_service_routes_concurrent_requests_through_lane(tmp_path):
    from assistant.models.mlx_engine import MlxEnginePool
    from assistant.models.mlx_service import MlxModelService

    engine = FakeEngine()
    # Preset lane: lane_for returns an existing lane without probing mlx-lm, so this
    # exercises the full service wiring (queue protocol, marker state machine, usage event).
    engine._batch_lane = BatchLane(
        engine, 4, generator_factory=lambda e, c: FakeBatchGen([[10, 11, EOS]])
    )
    pool = MlxEnginePool(max_loaded=2, loader=lambda path, _k=None: engine)
    svc = MlxModelService(
        models_dir=tmp_path, include_hf_cache=False, pool=pool, available_override=True
    )
    _make_model(tmp_path, "m")
    await svc.start()
    events = [
        ev
        async for ev in svc.stream_chat(
            [{"role": "user", "content": "hi"}], "m", concurrent=True
        )
    ]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "<10><11>"  # FakeEngine has no stream_text — only the lane can produce this
    usage = [e for e in events if e["type"] == "usage"]
    assert usage and usage[0]["output_tokens"] == 2


async def test_concurrent_flag_falls_back_to_serial_when_lane_unavailable(tmp_path):
    from assistant.models.mlx_engine import MlxEnginePool
    from assistant.models.mlx_service import MlxModelService

    class SerialOnly:  # the classic fake engine shape: stream_text, nothing batchable
        def stream_text(self, messages, **kwargs):
            yield "Hel"
            yield "lo"

    pool = MlxEnginePool(max_loaded=2, loader=lambda path, _k=None: SerialOnly())
    svc = MlxModelService(
        models_dir=tmp_path, include_hf_cache=False, pool=pool, available_override=True
    )
    _make_model(tmp_path, "m")
    await svc.start()
    events = [
        ev
        async for ev in svc.stream_chat(
            [{"role": "user", "content": "hi"}], "m", concurrent=True
        )
    ]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "Hello"


async def test_omlx_service_strips_concurrent_flag():
    from assistant.models.service import OmlxModelService

    captured = {}

    class StubClient:
        def stream_chat(self, messages, model, tools=None, **params):
            captured.update(params)

            async def gen():
                yield {"type": "text", "content": "ok"}

            return gen()

    svc = OmlxModelService(StubClient(), process=None)
    events = [
        ev
        async for ev in svc.stream_chat(
            [{"role": "user", "content": "x"}], "m", concurrent=True, max_tokens=5
        )
    ]
    assert events == [{"type": "text", "content": "ok"}]
    assert "concurrent" not in captured
    assert captured["max_tokens"] == 5


def test_prompt_cache_store_trims_and_evicts():
    trims = []
    store = PromptCacheStore(max_entries=2, trimmer=lambda c, d: trims.append((c, d)) or d)
    store.put([1, 2, 3, 4], "A")
    # Partial overlap: entry holds 4 ids, request shares 2 → trim 2 off before reuse.
    suffix, cache = store.take([1, 2, 9, 9])
    assert (suffix, cache) == ([9, 9], "A")
    assert trims == [("A", 2)]
    assert store.take([1, 2, 9, 9]) == ([1, 2, 9, 9], None)  # taken = gone until re-banked
    store.put([1], "B")
    store.put([2], "C")
    store.put([3], "D")  # cap 2 → "B" (oldest) evicted
    assert store.take([1, 5])[1] is None
    # A full-prompt match must still leave one token to feed the generator.
    store.put([7, 8], "E")
    suffix, cache = store.take([7, 8])
    assert suffix == [8] and cache == "E"


def test_untrimmable_cache_falls_back_to_full_prefill():
    store = PromptCacheStore(trimmer=lambda cache, drop: 0)  # all-or-nothing refusal
    store.put([1, 2, 3], "A")
    assert store.take([1, 2, 9]) == ([1, 2, 9], None)


def test_lane_for_rejects_unbatchable_engines():
    class Plain:  # no encode_prompt/_model — the serial-only FakeEngine shape
        pass

    engine = Plain()
    assert lane_for(engine, 8) is None
    assert engine._batch_unsupported is True  # verdict cached, no re-probe per turn
    assert lane_for(FakeEngine(), 0) is None  # concurrency 0 = lane disabled
