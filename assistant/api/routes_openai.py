"""OpenAI-compatible shim: ``GET /v1/models`` + ``POST /v1/chat/completions``.

Raw passthrough to the local model service (see compat.py) so OpenAI-style clients can use the
local MLX models. The model service already speaks the OpenAI message/tool shape, so this is
mostly framing the ``{"type": "text"|"tool_calls"}`` event stream as OpenAI completion chunks.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from assistant.api.compat import resolve_model, sampling_params

router = APIRouter(tags=["openai-compat"])


def _oa_tool_calls(calls: list[dict]) -> list[dict]:
    """Model-service tool calls → OpenAI ``tool_calls`` (arguments as a JSON string)."""
    return [
        {
            "id": c.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": c.get("name", ""),
                "arguments": json.dumps(c.get("arguments", {}), ensure_ascii=False),
            },
        }
        for c in calls
    ]


@router.get("/v1/models")
async def list_models(request: Request):
    service = request.app.state.model_service
    models = await service.list_models()
    return {
        "object": "list",
        "data": [
            {"id": m.id, "object": "model", "created": 0, "owned_by": m.source}
            for m in models
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    service = request.app.state.model_service
    body = await request.json()
    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="'messages' is required")
    model = await resolve_model(service, body.get("model", ""))
    tools = body.get("tools")
    params = sampling_params(body)
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if not body.get("stream"):
        text_parts: list[str] = []
        calls: list[dict] = []
        async for ev in service.stream_chat(messages, model, tools=tools, **params):
            if ev["type"] == "text":
                text_parts.append(ev["content"])
            elif ev["type"] == "tool_calls":
                calls.extend(ev["tool_calls"])
        message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
        if calls:
            message["tool_calls"] = _oa_tool_calls(calls)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if calls else "stop",
            }],
        }

    async def event_stream():
        def chunk(delta: dict, finish=None) -> str:
            payload = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield chunk({"role": "assistant"})
        finish = "stop"
        try:
            async for ev in service.stream_chat(messages, model, tools=tools, **params):
                if ev["type"] == "text":
                    yield chunk({"content": ev["content"]})
                elif ev["type"] == "tool_calls":
                    finish = "tool_calls"
                    tcs = _oa_tool_calls(ev["tool_calls"])
                    for i, tc in enumerate(tcs):
                        yield chunk({"tool_calls": [{"index": i, **tc}]})
        except Exception as exc:  # surface as a final error chunk, then close cleanly
            yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
            return
        yield chunk({}, finish=finish)
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
