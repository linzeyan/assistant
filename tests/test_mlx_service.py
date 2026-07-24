from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from assistant.models.mlx_engine import MlxEnginePool, ModelAdmissionError
from assistant.models.mlx_service import MlxModelService
from assistant.models.status import BackendState


class FakeEngine:
    def __init__(self, tokens):
        self._tokens = list(tokens)

    def stream_text(self, messages, **kwargs):
        yield from self._tokens


def _make_model(tmp_path: Path, name: str, *, arch: str = "LlamaForCausalLM") -> Path:
    import json

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (d / "model.safetensors").write_bytes(b"\x00")  # discovery requires real weights
    return d


def _service(tmp_path: Path, tokens=("Hel", "lo")):
    pool = MlxEnginePool(max_loaded=2, loader=lambda path, _k=None: FakeEngine(tokens))
    return MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=pool,
        available_override=True,
    )


async def test_type_override_reroutes_the_loader(tmp_path):
    # The per-model type override must reach the loader as forced_kind, so a checkpoint we'd
    # auto-detect one way can be forced to load the other way (the gemma-as-VLM crash fix).
    from assistant.models.per_model_store import PerModelStore

    _make_model(tmp_path, "qwen")  # auto-detects as an llm
    seen: list[str | None] = []

    def recording_loader(path, forced_kind=None):
        seen.append(forced_kind)
        return FakeEngine(("hi",))

    store = PerModelStore(tmp_path / "pm.json")
    store.set("qwen", {"type": "vlm"})  # force it the other way
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=MlxEnginePool(max_loaded=2, loader=recording_loader),
        per_model=store,
        available_override=True,
    )
    await svc.start()
    await svc.load("qwen")
    assert seen == ["vlm"]  # the override, not the auto-detected "llm", reached the loader


async def test_type_override_shows_in_list_models(tmp_path):
    from assistant.models.per_model_store import PerModelStore

    _make_model(tmp_path, "qwen")
    store = PerModelStore(tmp_path / "pm.json")
    store.set("qwen", {"type": "vlm"})
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=MlxEnginePool(max_loaded=2, loader=lambda p, _k=None: FakeEngine(("hi",))),
        per_model=store,
        available_override=True,
    )
    await svc.start()
    got = {m.id: m.type for m in await svc.list_models()}
    assert got["qwen"] == "vlm"  # reported under the overridden kind, not the detected one


async def test_per_model_template_kwargs_reach_the_engine(tmp_path):
    # 2-B: a model's saved chat_template_kwargs must arrive at stream_text merged with the
    # caller's — stored keys win (same semantics as sampler overrides), caller-only keys survive.
    from assistant.models.per_model_store import PerModelStore

    _make_model(tmp_path, "qwen")
    seen: list[dict] = []

    class RecordingEngine:
        def stream_text(self, messages, **kwargs):
            seen.append(kwargs.get("chat_template_kwargs"))
            yield "ok"

    store = PerModelStore(tmp_path / "pm.json")
    store.set("qwen", {"chat_template_kwargs": {"enable_thinking": True}})
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=MlxEnginePool(max_loaded=1, loader=lambda p, _k=None: RecordingEngine()),
        per_model=store,
        available_override=True,
    )
    await svc.start()
    # Caller passes its own kwargs (fusion's enable_thinking=False + an extra key): the stored
    # enable_thinking wins, the caller's other key survives the merge.
    [ev async for ev in svc.stream_chat(
        [{"role": "user", "content": "hi"}], "qwen",
        chat_template_kwargs={"enable_thinking": False, "other": 1},
    )]
    assert seen[-1] == {"enable_thinking": True, "other": 1}
    # No caller kwargs → the stored dict alone.
    [ev async for ev in svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")]
    assert seen[-1] == {"enable_thinking": True}


async def test_generation_runs_on_the_models_load_thread(tmp_path):
    # The thread-affinity contract behind the mlx-vlm "There is no Stream(gpu, N) in current
    # thread" crash: a model must generate on the exact thread it was loaded on. The service
    # must therefore submit the streaming worker to the pool's per-model executor, never to
    # the shared default executor.
    import threading

    idents: dict[str, int] = {}

    class ThreadRecordingEngine:
        def stream_text(self, messages, **kwargs):
            idents["generate"] = threading.get_ident()
            yield "ok"

    def loader(path, forced_kind=None):
        idents["load"] = threading.get_ident()
        return ThreadRecordingEngine()

    _make_model(tmp_path, "qwen")
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=MlxEnginePool(max_loaded=1, loader=loader),
        available_override=True,
    )
    await svc.start()
    out = [ev async for ev in svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")]
    assert any(ev.get("type") == "text" for ev in out)
    assert idents["generate"] == idents["load"]


def test_ram_ceiling_defaults_to_physical_memory(tmp_path):
    # "Check the resource fits before loading": with no explicit ceiling, the pool still gets one
    # derived from physical RAM, so an oversized model fails loud instead of OOM-crashing.
    svc = MlxModelService(models_dir=tmp_path, include_hf_cache=False, available_override=True)
    assert svc._pool._ceiling is not None and svc._pool._ceiling > 0


async def test_unavailable_without_mlx(tmp_path):
    svc = MlxModelService(
        models_dir=tmp_path, include_hf_cache=False, available_override=False
    )
    status = await svc.start()
    assert status.state == BackendState.UNAVAILABLE
    # Fail soft: no backend -> empty list, never an exception.
    assert await svc.list_models() == []


async def test_start_reports_local_with_discovery(tmp_path):
    _make_model(tmp_path, "qwen")
    svc = _service(tmp_path)
    status = await svc.start()
    assert status.state == BackendState.LOCAL
    assert "1 models" in status.detail


async def test_list_models_reflects_loaded_state(tmp_path):
    _make_model(tmp_path, "qwen")
    svc = _service(tmp_path)
    await svc.start()

    models = await svc.list_models()
    assert [m.id for m in models] == ["qwen"]
    assert models[0].loaded is False
    assert models[0].source == "local"

    await svc.load("qwen")
    assert (await svc.list_models())[0].loaded is True

    await svc.unload("qwen")
    assert (await svc.list_models())[0].loaded is False


async def test_stream_chat_yields_text_tokens(tmp_path):
    _make_model(tmp_path, "qwen")
    svc = _service(tmp_path, tokens=("Hel", "lo"))
    await svc.start()

    events = [
        e async for e in svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")
    ]
    assert all(e["type"] == "text" for e in events)
    assert "".join(e["content"] for e in events) == "Hello"
    # The model got loaded into the pool as a side effect of streaming.
    assert svc._pool.is_loaded("qwen")


async def test_load_unknown_model_raises(tmp_path):
    svc = _service(tmp_path)
    await svc.start()
    with pytest.raises(ValueError):
        await svc.load("does-not-exist")


async def test_unload_frees_engine_not_pinned_by_stream_frame(tmp_path):
    # Regression: after a chat turn the _stream() async-generator frame must not keep the
    # engine alive, or pressing Unload pops the pool yet gc can't reclaim the model weights
    # (unified memory never returns). Reproduce with a still-open generator and assert the
    # engine becomes collectible once the pool drops it.
    import asyncio
    import gc
    import weakref

    _make_model(tmp_path, "qwen")
    engine = FakeEngine(("hi",))
    ref = weakref.ref(engine)
    # The loader closes over `engine` and only runs before the `del engine` below; ruff sees the
    # later del and flags it as maybe-unbound (false positive for this ordering).
    pool = MlxEnginePool(max_loaded=1, loader=lambda _path, _k=None: engine)  # noqa: F821
    svc = MlxModelService(
        models_dir=tmp_path, include_hf_cache=False, pool=pool, available_override=True
    )
    await svc.start()

    gen = svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")
    await gen.__anext__()  # pull one event; the generator is now suspended mid-turn
    await svc.unload("qwen")  # drop the pool's reference
    del engine  # drop the test's own reference

    # The worker thread finishes its single token quickly; once it does, only a leaked
    # frame reference could keep the engine alive. Poll briefly to avoid thread-timing
    # flakiness; with the leak it never collects and this assertion fails.
    collected = False
    for _ in range(50):
        gc.collect()
        if ref() is None:
            collected = True
            break
        await asyncio.sleep(0.02)
    await gen.aclose()
    assert collected, "engine was pinned by the _stream frame after unload"


async def test_load_rejects_non_chat_model(tmp_path):
    # An embedding / text-encoder is discovered and listed (so the user sees it) but
    # can't be a chat model. Loading must fail loud, not crash deep in the engine.
    _make_model(tmp_path, "encoder", arch="UMT5EncoderModel")
    svc = _service(tmp_path)
    await svc.start()

    models = await svc.list_models()
    assert models[0].type == "embedding"  # surfaced to the GUI so it disables Load/Use

    with pytest.raises(ValueError, match="chat model"):
        await svc.load("encoder")


async def test_load_allows_vlm_model(tmp_path):
    # VL / omni checkpoints route through mlx-vlm, so they ARE chattable. (The fake pool
    # loader stands in for mlx-vlm here; kind classification is what gates the load.)
    _make_model(tmp_path, "omni", arch="Qwen2_5_VLForConditionalGeneration")
    svc = _service(tmp_path)
    await svc.start()

    models = await svc.list_models()
    assert models[0].type == "vlm"

    await svc.load("omni")
    assert svc._pool.is_loaded("omni")


_READ_FILE_TOOL = [
    {"type": "function", "function": {"name": "read_file", "parameters": {}}}
]


async def test_stream_chat_emits_structured_tool_calls(tmp_path):
    _make_model(tmp_path, "qwen")
    tokens = (
        "<tool_call>",
        '{"name": "read_file", "arguments": {"path": "a.py"}}',
        "</tool_call>",
    )
    svc = _service(tmp_path, tokens=tokens)
    await svc.start()

    events = [
        e
        async for e in svc.stream_chat(
            [{"role": "user", "content": "read a.py"}], "qwen", tools=_READ_FILE_TOOL
        )
    ]
    tool_events = [e for e in events if e["type"] == "tool_calls"]
    assert len(tool_events) == 1
    (call,) = tool_events[0]["tool_calls"]
    # Ids are minted uniquely per call (repeating "call_0" made Anthropic-protocol
    # clients drop every tool call after the first turn), so match on shape not value.
    assert call["id"].startswith("call_")
    assert call["name"] == "read_file"
    assert call["arguments"] == {"path": "a.py"}
    # The raw <tool_call> markup must never leak into streamed text.
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert "<tool_call>" not in text


async def test_stream_chat_emits_prose_before_suppressing_tool_markup(tmp_path):
    _make_model(tmp_path, "qwen")
    tokens = (
        "Reading the file now. ",
        "<tool_call>",
        '{"name": "read_file", "arguments": {"path": "a.py"}}',
        "</tool_call>",
    )
    svc = _service(tmp_path, tokens=tokens)
    await svc.start()

    events = [
        e
        async for e in svc.stream_chat(
            [{"role": "user", "content": "x"}], "qwen", tools=_READ_FILE_TOOL
        )
    ]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "Reading the file now. "
    assert any(e["type"] == "tool_calls" for e in events)


async def test_stream_chat_drops_truncated_tool_marker_instead_of_leaking_it(tmp_path):
    # The user's screenshot bug: a turn ended with a bare "<tool_call>" (the model opened a tool
    # call then hit EOS/max_tokens with nothing inside). parse returns no call, and the raw marker
    # must NOT leak back as visible text — once a marker opens, the tail is a failed tool attempt.
    _make_model(tmp_path, "qwen")
    tokens = ("All done. ", "<tool_call>")  # opened, never filled
    svc = _service(tmp_path, tokens=tokens)
    await svc.start()

    events = [
        e
        async for e in svc.stream_chat(
            [{"role": "user", "content": "x"}], "qwen", tools=_READ_FILE_TOOL
        )
    ]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "All done. "  # prose kept, bare marker dropped
    assert "<tool_call>" not in text
    assert not any(e["type"] == "tool_calls" for e in events)


_EDIT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "parameters": {"properties": {"replace_all": {"type": "boolean"}}},
        },
    }
]


async def test_stream_chat_normalizes_overquoted_scalar_against_schema(tmp_path):
    # Qwen3-Coder emitted `"replace_all": "False"` (string); the middleware must coerce it to the
    # boolean the schema declares before the event leaves the service, or the downstream validator
    # rejects the whole call (the observed InputValidationError).
    _make_model(tmp_path, "qwen")
    tokens = (
        "<tool_call>",
        '{"name": "Edit", "arguments": {"replace_all": "False", "file_path": "a.md"}}',
        "</tool_call>",
    )
    svc = _service(tmp_path, tokens=tokens)
    await svc.start()

    events = [
        e
        async for e in svc.stream_chat(
            [{"role": "user", "content": "x"}], "qwen", tools=_EDIT_TOOL
        )
    ]
    (tc,) = [e for e in events if e["type"] == "tool_calls"][0]["tool_calls"]
    assert tc["arguments"]["replace_all"] is False  # coerced string -> bool
    assert tc["arguments"]["file_path"] == "a.md"  # untouched


async def test_stream_chat_plain_text_has_no_tool_calls(tmp_path):
    _make_model(tmp_path, "qwen")
    svc = _service(tmp_path, tokens=("The answer ", "is 42."))
    await svc.start()

    events = [
        e async for e in svc.stream_chat([{"role": "user", "content": "x"}], "qwen")
    ]
    assert all(e["type"] == "text" for e in events)
    assert "".join(e["content"] for e in events) == "The answer is 42."


def test_vlm_chat_engine_streams_text(monkeypatch):
    # VlmChatEngine renders the prompt via the processor's chat template and streams
    # text out of mlx-vlm. Verify the wiring with a fake processor + stream_generate,
    # so no real VLM is loaded. Skips where mlx-vlm isn't installed (e.g. CI).
    pytest.importorskip("mlx_vlm", exc_type=ImportError)
    import mlx_vlm

    from assistant.models.mlx_engine import VlmChatEngine

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kw):
            assert messages[-1]["content"] == "hi"
            return "PROMPT"

    class Chunk:
        def __init__(self, text):
            self.text = text

    monkeypatch.setattr(
        mlx_vlm, "stream_generate", lambda *a, **k: iter([Chunk("Hel"), Chunk("lo")])
    )
    engine = VlmChatEngine(object(), FakeProcessor())
    out = "".join(engine.stream_text([{"role": "user", "content": "hi"}]))
    assert out == "Hello"


# --- C: memory-aware admission (byte-level ceiling on top of the count cap) ---------------

def _sized_model(tmp_path: Path, name: str, nbytes: int) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "model.safetensors").write_bytes(b"\x00" * nbytes)  # _estimate reads this file's size
    return d


async def test_pool_evicts_to_stay_under_memory_ceiling(tmp_path, monkeypatch):
    # Byte-admission: even when the count cap wouldn't trigger, the ceiling alone evicts the LRU
    # model so the live set stays within budget. Pin the footprint to the disk estimate (no live
    # MLX measurement) for a deterministic check.
    monkeypatch.setattr("assistant.models.mlx_engine._active_memory_bytes", lambda: None)
    a = _sized_model(tmp_path, "a", 100)
    b = _sized_model(tmp_path, "b", 100)
    c = _sized_model(tmp_path, "c", 100)
    pool = MlxEnginePool(max_loaded=10, mem_ceiling_bytes=250, loader=lambda _p, _k=None: object())
    await pool.acquire("a", a)
    await pool.acquire("b", b)
    await pool.acquire("c", c)  # a+b+c = 300 > 250 → evict LRU "a"
    assert pool.loaded_ids() == ["b", "c"]


async def test_pool_rejects_model_larger_than_ceiling(tmp_path, monkeypatch):
    # A single model bigger than the whole budget fails loud BEFORE loading — never OOM-loaded.
    monkeypatch.setattr("assistant.models.mlx_engine._active_memory_bytes", lambda: None)
    big = _sized_model(tmp_path, "big", 300)
    pool = MlxEnginePool(max_loaded=10, mem_ceiling_bytes=250, loader=lambda _p, _k=None: object())
    with pytest.raises(ModelAdmissionError):
        await pool.acquire("big", big)
    assert pool.loaded_ids() == []  # nothing loaded


async def test_pool_rejects_when_pinned_models_leave_no_room(tmp_path, monkeypatch):
    # A pinned model's footprint counts against the budget and can't be evicted, so an incoming
    # model that won't fit in the remainder is rejected rather than OOM-loaded over the pin.
    monkeypatch.setattr("assistant.models.mlx_engine._active_memory_bytes", lambda: None)
    a = _sized_model(tmp_path, "a", 100)
    b = _sized_model(tmp_path, "b", 100)
    pool = MlxEnginePool(max_loaded=10, mem_ceiling_bytes=150, loader=lambda _p, _k=None: object())
    await pool.acquire("a", a)
    pool.pin("a")  # 100 held, only 50 free under the 150 ceiling
    with pytest.raises(ModelAdmissionError):
        await pool.acquire("b", b)  # needs 100 > 50, can't evict pinned "a"
    assert pool.loaded_ids() == ["a"]  # the pinned model survived


async def test_pool_without_ceiling_keeps_count_only_behaviour(tmp_path, monkeypatch):
    # No ceiling → byte-admission disabled: the pool holds up to max_loaded regardless of size,
    # exactly as before this feature (opt-in guardrail).
    monkeypatch.setattr("assistant.models.mlx_engine._active_memory_bytes", lambda: None)
    a = _sized_model(tmp_path, "a", 10_000)
    b = _sized_model(tmp_path, "b", 10_000)
    pool = MlxEnginePool(max_loaded=2, mem_ceiling_bytes=None, loader=lambda _p, _k=None: object())
    await pool.acquire("a", a)
    await pool.acquire("b", b)
    assert pool.loaded_ids() == ["a", "b"]  # both held; no rejection despite large sizes


async def test_client_disconnect_stops_engine_generation(tmp_path):
    # The N81 zombie: the engine generator runs on an executor thread nothing can
    # interrupt, so after a disconnect it used to decode to max_tokens — burning the
    # model's single engine slot while later requests queued behind it. Closing the
    # stream must reach the worker via the stop flag within ~one token.
    import threading
    import time

    class SlowEngine:
        def __init__(self):
            self.consumed = 0
            self.closed = threading.Event()

        def stream_text(self, messages, **kwargs):
            try:
                for i in range(2000):
                    self.consumed = i
                    time.sleep(0.005)
                    yield f"token {i} "
            finally:
                self.closed.set()

    _make_model(tmp_path, "qwen")
    eng = SlowEngine()
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=MlxEnginePool(max_loaded=2, loader=lambda path, _k=None: eng),
        available_override=True,
    )
    gen = svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")
    assert (await gen.__anext__())["type"] == "text"  # generation is live
    await gen.aclose()  # client disconnects
    for _ in range(400):  # worker must wind down promptly, not run all 2000 tokens
        if eng.closed.is_set():
            break
        await asyncio.sleep(0.01)
    assert eng.closed.is_set()
    assert eng.consumed < 1900


HARMONY_STEP = (
    "<|channel|>analysis<|message|>Need to search first.<|end|>"
    "<|start|>assistant<|channel|>commentary to=functions.web_search "
    '<|constrain|>json<|message|>{"query": "x"}'
)


async def test_harmony_stream_sanitizes_channel_markup(tmp_path):
    # N84: raw <|channel|> markup must never reach the client. Reasoning is re-shaped
    # into the product's <think> convention (the GUI collapses it, N1) and the call
    # arrives structurally — same contract every other tool format already honors.
    _make_model(tmp_path, "qwen")
    svc = _service(tmp_path, tokens=(HARMONY_STEP[:40], HARMONY_STEP[40:]))
    events = [
        e async for e in svc.stream_chat(
            [{"role": "user", "content": "hi"}], "qwen",
            tools=[{"function": {"name": "web_search", "parameters": {}}}],
        )
    ]
    texts = [e["content"] for e in events if e["type"] == "text"]
    assert texts == ["<think>Need to search first.</think>"]
    calls = [e for e in events if e["type"] == "tool_calls"]
    assert len(calls) == 1
    assert calls[0]["tool_calls"][0]["name"] == "web_search"


async def test_harmony_final_answer_emits_think_and_prose(tmp_path):
    _make_model(tmp_path, "qwen")
    final_step = (
        "<|channel|>analysis<|message|>Sum up.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Python 3.13.14 is the latest."
    )
    svc = _service(tmp_path, tokens=(final_step,))
    events = [
        e async for e in svc.stream_chat([{"role": "user", "content": "hi"}], "qwen")
    ]
    texts = [e["content"] for e in events if e["type"] == "text"]
    assert texts == ["<think>Sum up.</think>\n\nPython 3.13.14 is the latest."]
    assert not [e for e in events if e["type"] == "tool_calls"]
