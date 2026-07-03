"""OpenAI- and Anthropic-compatible shims: translation + endpoint framing.

These let external agents (Claude Code via ANTHROPIC_BASE_URL, OpenAI clients) drive the local
models as a raw chat backend. The translation is pure and unit-tested; the endpoints are checked
against a fake model service that records what it received (so we know the translation reached it)
and scripts text + tool-call events back.
"""

from __future__ import annotations

import json

from assistant.api.compat import (
    anthropic_to_openai_messages,
    anthropic_tools_to_openai,
    resolve_model,
    sampling_params,
)
from assistant.config import Settings
from assistant.main import create_app
from assistant.models.types import ModelInfo
from fastapi.testclient import TestClient


class _FakeService:
    """Records the (messages, model, tools) it was called with; scripts events back."""

    def __init__(self, events=None, models=None, count=None):
        self._events = events or [{"type": "text", "content": "hello"}]
        self._models = models or [
            ModelInfo(id="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit", type="llm",
                      loaded=False, source="local", size_bytes=1),
            ModelInfo(id="mlx-community/gemma-4-31b-it-4bit", type="llm", loaded=False,
                      source="local", size_bytes=1),
        ]
        self._count = count
        self.seen: dict = {}
        self.seen_count: dict | None = None

    async def reachable(self):
        return True

    async def list_models(self):
        return self._models

    async def count_tokens(self, messages, model, tools=None):
        self.seen_count = {"messages": messages, "model": model, "tools": tools}
        return self._count

    def stream_chat(self, messages, model, tools=None, **params):
        self.seen = {"messages": messages, "model": model, "tools": tools, "params": params}

        async def gen():
            for e in self._events:
                yield e

        return gen()


def _client(tmp_path):
    # NB: set app.state.model_service INSIDE the `with` block — startup (on context enter)
    # installs the real service, so an override before entering would be clobbered.
    return TestClient(create_app(Settings(sessions_dir=tmp_path / "s", models_dir=tmp_path / "m")))


# --- pure translation ---


def test_anthropic_to_openai_messages_system_and_tool_roundtrip():
    msgs = anthropic_to_openai_messages(
        system="be terse",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me check"},
                {"type": "tool_use", "id": "tu1", "name": "read", "input": {"path": "x"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "content": "FILE BODY"},
            ]},
        ],
    )
    assert msgs[0] == {"role": "system", "content": "be terse"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    asst = msgs[2]
    assert asst["role"] == "assistant" and asst["tool_calls"][0]["function"]["name"] == "read"
    assert json.loads(asst["tool_calls"][0]["function"]["arguments"]) == {"path": "x"}
    assert msgs[3] == {"role": "tool", "tool_call_id": "tu1", "content": "FILE BODY"}


def test_anthropic_tools_to_openai_schema():
    out = anthropic_tools_to_openai([
        {"name": "read", "description": "read a file",
         "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}},
    ])
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "read"
    assert out[0]["function"]["parameters"]["properties"]["path"]["type"] == "string"
    assert anthropic_tools_to_openai(None) is None


async def test_resolve_model_matches_basename_then_substring():
    svc = _FakeService()
    # Claude Code sends the short id without the mlx-community/ prefix.
    assert await resolve_model(svc, "Qwen3-Coder-30B-A3B-Instruct-8bit") == \
        "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
    assert await resolve_model(svc, "gemma-4") == "mlx-community/gemma-4-31b-it-4bit"
    assert await resolve_model(svc, "nope") == "nope"  # unknown falls through unchanged


def test_sampling_params_omits_none():
    assert sampling_params({"max_tokens": 50, "temperature": 0.2, "top_p": None}) == \
        {"max_tokens": 50, "temperature": 0.2}


# --- OpenAI endpoint ---


def test_openai_chat_completion_nonstream_with_tool_call(tmp_path):
    svc = _FakeService(events=[
        {"type": "text", "content": "sure"},
        {"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "read", "arguments": {"p": 1}}]},
    ])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/chat/completions", json={
            "model": "gemma-4", "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
        })
    body = r.json()
    assert r.status_code == 200 and body["object"] == "chat.completion"
    assert svc.seen["model"] == "mlx-community/gemma-4-31b-it-4bit"  # resolved
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tc = choice["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "read" and json.loads(tc["function"]["arguments"]) == {"p": 1}


def test_openai_chat_completion_streams_chunks(tmp_path):
    svc = _FakeService(events=[{"type": "text", "content": "hi there"}])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/chat/completions", json={
            "model": "gemma-4", "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert "data: [DONE]" in r.text
    deltas = [json.loads(ln[6:])["choices"][0]["delta"]
              for ln in r.text.splitlines() if ln.startswith("data: ") and "[DONE]" not in ln]
    assert any(d.get("content") == "hi there" for d in deltas)


# --- Anthropic endpoint ---


def test_anthropic_messages_nonstream_with_tool_use(tmp_path):
    svc = _FakeService(events=[
        {"type": "text", "content": "let me read it"},
        {"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "read", "arguments": {"p": 1}}]},
    ])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages", json={
            "model": "Qwen3-Coder-30B-A3B-Instruct-8bit", "max_tokens": 100,
            "system": "be terse", "messages": [{"role": "user", "content": "read x"}],
            "tools": [{"name": "read", "input_schema": {"type": "object", "properties": {}}}],
        })
    body = r.json()
    assert r.status_code == 200 and body["type"] == "message"
    assert body["stop_reason"] == "tool_use"
    assert svc.seen["messages"][0] == {"role": "system", "content": "be terse"}  # system translated
    assert svc.seen["tools"][0]["function"]["name"] == "read"  # tools translated
    blocks = {b["type"]: b for b in body["content"]}
    assert blocks["text"]["text"] == "let me read it"
    assert blocks["tool_use"]["name"] == "read" and blocks["tool_use"]["input"] == {"p": 1}


def test_anthropic_messages_streams_event_sequence(tmp_path):
    svc = _FakeService(events=[{"type": "text", "content": "hello"}])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages", json={
            "model": "gemma-4", "max_tokens": 50, "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
    events = [ln[len("event: "):] for ln in r.text.splitlines() if ln.startswith("event: ")]
    # The Anthropic streaming contract Claude Code parses: start → block → delta → stop → end.
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert "content_block_start" in events and "content_block_delta" in events
    assert "message_delta" in events


def _sse_data(text: str) -> list[dict]:
    return [json.loads(ln[len("data: "):])
            for ln in text.splitlines() if ln.startswith("data: ")]


def test_anthropic_nonstream_reports_real_usage(tmp_path):
    # A: the compat route must surface the engine's token counts (was hardcoded 0/0), so Claude
    # Code can track context fill. The usage event never leaks into content blocks.
    svc = _FakeService(events=[
        {"type": "text", "content": "hi"},
        {"type": "usage", "input_tokens": 1234, "output_tokens": 7},
    ])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages", json={
            "model": "gemma-4", "max_tokens": 50,
            "messages": [{"role": "user", "content": "hi"}],
        })
    body = r.json()
    assert body["usage"] == {"input_tokens": 1234, "output_tokens": 7}
    assert [b["type"] for b in body["content"]] == ["text"]  # no stray "usage" block


def test_anthropic_stream_reports_usage_from_count_and_event(tmp_path):
    # A (streaming): input_tokens comes from count_tokens at message_start (known up front so Claude
    # Code sizes the window); output_tokens comes from the usage event at message_delta.
    svc = _FakeService(count=8000, events=[
        {"type": "text", "content": "hello"},
        {"type": "usage", "input_tokens": 8000, "output_tokens": 3},
    ])
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages", json={
            "model": "gemma-4", "max_tokens": 50, "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        })
    payloads = _sse_data(r.text)
    start = next(p for p in payloads if p["type"] == "message_start")
    delta = next(p for p in payloads if p["type"] == "message_delta")
    assert start["message"]["usage"]["input_tokens"] == 8000
    assert delta["usage"]["output_tokens"] == 3
    # The usage event must not surface as a text delta.
    assert all("usage" not in json.dumps(p.get("delta", {})) for p in payloads
               if p["type"] == "content_block_delta")


def test_count_tokens_endpoint_translates_and_returns_count(tmp_path):
    # C: /v1/messages/count_tokens renders via the translation layer and returns the model count.
    svc = _FakeService(count=4242)
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages/count_tokens", json={
            "model": "Qwen3-Coder-30B-A3B-Instruct-8bit",
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "read", "input_schema": {"type": "object", "properties": {}}}],
        })
    assert r.status_code == 200 and r.json() == {"input_tokens": 4242}
    assert svc.seen_count["model"] == "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
    assert svc.seen_count["messages"][0] == {"role": "system", "content": "be terse"}
    assert svc.seen_count["tools"][0]["function"]["name"] == "read"


def test_count_tokens_endpoint_zero_when_backend_cannot_count(tmp_path):
    # None from the backend (can't count) → 0, never a crash.
    svc = _FakeService(count=None)
    with _client(tmp_path) as client:
        client.app.state.model_service = svc
        r = client.post("/v1/messages/count_tokens", json={
            "model": "gemma-4", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.json() == {"input_tokens": 0}


# --- K4: errors must wear the Anthropic envelope so Claude Code can read error.type ---

def _assert_anthropic_error(body: dict, err_type: str):
    # The shape Claude Code parses: {"type":"error","error":{"type":..,"message":..}} — NOT the
    # bare FastAPI {"detail": ...}, which it can't interpret.
    assert body.get("type") == "error"
    assert body["error"]["type"] == err_type
    assert isinstance(body["error"].get("message"), str) and body["error"]["message"]
    assert "detail" not in body


def test_anthropic_missing_messages_is_shaped_400(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.model_service = _FakeService()
        r = client.post("/v1/messages", json={"model": "gemma-4", "max_tokens": 10})
    assert r.status_code == 400
    _assert_anthropic_error(r.json(), "invalid_request_error")


def test_count_tokens_missing_messages_is_shaped_400(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.model_service = _FakeService()
        r = client.post("/v1/messages/count_tokens", json={"model": "gemma-4"})
    assert r.status_code == 400
    _assert_anthropic_error(r.json(), "invalid_request_error")


def test_anthropic_malformed_json_is_shaped_400(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.model_service = _FakeService()
        r = client.post("/v1/messages", content=b"{not json",
                        headers={"content-type": "application/json"})
    assert r.status_code == 400
    _assert_anthropic_error(r.json(), "invalid_request_error")


class _ExplodingService(_FakeService):
    """Model service whose generation blows up mid-turn (model won't load / template render)."""

    def stream_chat(self, messages, model, tools=None, **params):
        async def gen():
            raise RuntimeError("model failed to load")
            yield  # pragma: no cover — marks this an async generator

        return gen()


def test_anthropic_nonstream_generation_error_is_shaped_api_error(tmp_path):
    # A generation failure on the non-streaming path was a bare 500; now it's a shaped api_error,
    # mirroring the streaming path's in-band error event.
    with _client(tmp_path) as client:
        client.app.state.model_service = _ExplodingService()
        r = client.post("/v1/messages", json={
            "model": "gemma-4", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 500
    body = r.json()
    _assert_anthropic_error(body, "api_error")
    assert "model failed to load" in body["error"]["message"]


def test_anthropic_stream_error_event_keeps_envelope(tmp_path):
    # The in-stream failure event stays in the Anthropic error shape (shared _err_body).
    with _client(tmp_path) as client:
        client.app.state.model_service = _ExplodingService()
        r = client.post("/v1/messages", json={
            "model": "gemma-4", "max_tokens": 10, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]})
    err = next(d for d in _sse_data(r.text) if d.get("type") == "error")
    _assert_anthropic_error(err, "api_error")
