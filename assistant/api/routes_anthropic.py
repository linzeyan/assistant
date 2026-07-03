"""Anthropic-compatible shim: ``POST /v1/messages``.

This is what makes Claude Code (ANTHROPIC_BASE_URL) drive the local MLX models. Raw passthrough
to the model service (see compat.py): the Anthropic request (top-level ``system``, content blocks,
tool_use/tool_result) is translated to the service's OpenAI-ish shape, and the service's
``{"type": "text"|"tool_calls"}`` events are framed back as Anthropic message/content-block events
(streaming) or a single Messages response (non-streaming).
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from assistant.api.compat import (
    anthropic_to_openai_messages,
    anthropic_tools_to_openai,
    resolve_model,
    sampling_params,
)

router = APIRouter(tags=["anthropic-compat"])


def _evt(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/v1/messages")
async def messages(request: Request):
    service = request.app.state.model_service
    body = await request.json()
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="'messages' is required")
    model = await resolve_model(service, body.get("model", ""))
    oa_messages = anthropic_to_openai_messages(body.get("system"), body["messages"])
    tools = anthropic_tools_to_openai(body.get("tools"))
    params = sampling_params(body)
    if "max_tokens" not in params:  # Anthropic requires max_tokens; honour it as the cap
        params["max_tokens"] = body.get("max_tokens", 1024)
    mid = f"msg_{uuid.uuid4().hex}"

    if not body.get("stream"):
        text_parts: list[str] = []
        calls: list[dict] = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        async for ev in service.stream_chat(oa_messages, model, tools=tools, **params):
            if ev["type"] == "text":
                text_parts.append(ev["content"])
            elif ev["type"] == "tool_calls":
                calls.extend(ev["tool_calls"])
            elif ev["type"] == "usage":
                usage = {
                    "input_tokens": ev.get("input_tokens", 0),
                    "output_tokens": ev.get("output_tokens", 0),
                }
        content: list[dict] = []
        text = "".join(text_parts)
        if text:
            content.append({"type": "text", "text": text})
        for c in calls:
            content.append({
                "type": "tool_use",
                "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": c.get("name", ""),
                "input": c.get("arguments", {}),
            })
        return {
            "id": mid,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if calls else "end_turn",
            "stop_sequence": None,
            "usage": usage,
        }

    # Count the input up front so message_start carries the real context size — Claude Code reads
    # usage.input_tokens to know how full the window is and to trigger its own compaction. None
    # (backend can't count) → 0, the prior behaviour.
    try:
        input_tokens = await service.count_tokens(oa_messages, model, tools=tools) or 0
    except Exception:
        input_tokens = 0

    async def event_stream():
        yield _evt("message_start", {
            "type": "message_start",
            "message": {
                "id": mid, "type": "message", "role": "assistant", "model": model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            },
        })
        index = 0
        text_open = False
        stop_reason = "end_turn"
        output_tokens = 0
        try:
            async for ev in service.stream_chat(oa_messages, model, tools=tools, **params):
                if ev["type"] == "usage":
                    output_tokens = ev.get("output_tokens", output_tokens)
                    continue
                if ev["type"] == "text":
                    if not ev["content"]:
                        continue
                    if not text_open:
                        yield _evt("content_block_start", {
                            "type": "content_block_start", "index": index,
                            "content_block": {"type": "text", "text": ""},
                        })
                        text_open = True
                    yield _evt("content_block_delta", {
                        "type": "content_block_delta", "index": index,
                        "delta": {"type": "text_delta", "text": ev["content"]},
                    })
                elif ev["type"] == "tool_calls":
                    stop_reason = "tool_use"
                    if text_open:
                        yield _evt("content_block_stop",
                                   {"type": "content_block_stop", "index": index})
                        text_open = False
                        index += 1
                    for c in ev["tool_calls"]:
                        yield _evt("content_block_start", {
                            "type": "content_block_start", "index": index,
                            "content_block": {
                                "type": "tool_use",
                                "id": c.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": c.get("name", ""), "input": {},
                            },
                        })
                        yield _evt("content_block_delta", {
                            "type": "content_block_delta", "index": index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(
                                    c.get("arguments", {}), ensure_ascii=False),
                            },
                        })
                        yield _evt("content_block_stop",
                                   {"type": "content_block_stop", "index": index})
                        index += 1
        except Exception as exc:
            yield _evt("error", {"type": "error",
                                 "error": {"type": "api_error", "message": str(exc)}})
            return
        if text_open:
            yield _evt("content_block_stop", {"type": "content_block_stop", "index": index})
        yield _evt("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })
        yield _evt("message_stop", {"type": "message_stop"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    """Anthropic's token-counting endpoint. Claude Code calls it to size a conversation against the
    model's window before sending — without it, it can only guess. Returns the rendered prompt's
    token count (0 when the backend can't count)."""
    service = request.app.state.model_service
    body = await request.json()
    if not body.get("messages"):
        raise HTTPException(status_code=400, detail="'messages' is required")
    model = await resolve_model(service, body.get("model", ""))
    oa_messages = anthropic_to_openai_messages(body.get("system"), body["messages"])
    tools = anthropic_tools_to_openai(body.get("tools"))
    count = await service.count_tokens(oa_messages, model, tools=tools)
    return {"input_tokens": int(count or 0)}
