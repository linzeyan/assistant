"""Seam-contract tests for the model layer.

These mock omlx's HTTP surface with ``httpx.MockTransport`` so the contract is
enforced regardless of omlx's internals (plan Part E, risk 5). No network, no omlx
install required.
"""

from __future__ import annotations

import json

import httpx
import pytest

from assistant.models.omlx_client import OmlxClient
from assistant.models.omlx_subprocess import OmlxProcess, OmlxState
from assistant.models.service import OmlxModelService


def make_client(handler) -> OmlxClient:
    return OmlxClient("http://test", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_list_models_parses_v1_models_and_loaded_state():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {"id": "qwen3-8b", "type": "llm", "source": "local"},
                        {"id": "llava", "model_type": "vlm"},
                    ],
                },
            )
        if request.url.path == "/v1/models/status":
            return httpx.Response(200, json={"models": [{"id": "qwen3-8b", "loaded": True}]})
        return httpx.Response(404)

    client = make_client(handler)
    models = await client.list_models()
    assert {m.id for m in models} == {"qwen3-8b", "llava"}
    assert {m.id for m in models if m.loaded} == {"qwen3-8b"}
    # type/source mapping handles both naming variants omlx may emit.
    by_id = {m.id: m for m in models}
    assert by_id["llava"].type == "vlm"
    assert by_id["qwen3-8b"].source == "local"
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_yields_text_events():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["model"] == "qwen3-8b"
        sse = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    events = [
        e
        async for e in client.stream_chat([{"role": "user", "content": "hi"}], "qwen3-8b")
    ]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "Hello"
    assert all(e["type"] == "text" for e in events)  # no tool_calls event when none
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_chat_assembles_streamed_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        # arguments arrive split across two deltas; id/name only in the first.
        sse = (
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            '"function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}\n\n'
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"th\\":\\"a.txt\\"}"}}]}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    client = make_client(handler)
    events = [
        e async for e in client.stream_chat([{"role": "user", "content": "hi"}], "m")
    ]
    tool_events = [e for e in events if e["type"] == "tool_calls"]
    assert len(tool_events) == 1
    calls = tool_events[0]["tool_calls"]
    assert calls == [
        {"id": "call_1", "name": "read_file", "arguments": {"path": "a.txt"}}
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_service_degrades_gracefully_when_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = make_client(handler)
    process = OmlxProcess(
        client,
        host="127.0.0.1",
        port=8000,
        models_dir=None,
        omlx_bin="/nonexistent/omlx",  # forces the "not found" path
        autostart=True,
    )
    service = OmlxModelService(client, process)

    # Unreachable backend -> empty list, never an exception.
    assert await service.list_models() == []
    # ensure_running with a missing binary -> UNAVAILABLE status, not a crash.
    status = await service.start()
    assert status.state == OmlxState.UNAVAILABLE
    assert "brew install omlx" in status.detail
    await client.aclose()
