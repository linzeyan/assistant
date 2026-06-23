from __future__ import annotations

from pathlib import Path

import pytest

from assistant.models.mlx_engine import MlxEnginePool
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
    pool = MlxEnginePool(max_loaded=2, loader=lambda path: FakeEngine(tokens))
    return MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=pool,
        available_override=True,
    )


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
    pool = MlxEnginePool(max_loaded=1, loader=lambda _path: engine)
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
    assert tool_events[0]["tool_calls"] == [
        {"id": "call_0", "name": "read_file", "arguments": {"path": "a.py"}}
    ]
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
    pytest.importorskip("mlx_vlm")
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
