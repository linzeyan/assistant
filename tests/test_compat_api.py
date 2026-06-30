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

    def __init__(self, events=None, models=None):
        self._events = events or [{"type": "text", "content": "hello"}]
        self._models = models or [
            ModelInfo(id="mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit", type="llm",
                      loaded=False, source="local", size_bytes=1),
            ModelInfo(id="mlx-community/gemma-4-31b-it-4bit", type="llm", loaded=False,
                      source="local", size_bytes=1),
        ]
        self.seen: dict = {}

    async def reachable(self):
        return True

    async def list_models(self):
        return self._models

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
