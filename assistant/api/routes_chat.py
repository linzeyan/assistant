from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from assistant.tools.approval import InteractiveApprover, resolve_pending

router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: str | None = None
    # When set, approval-required tools pause and stream an ``approval_request``
    # event; the client decides via POST /chat/approve. The desktop GUI sets this.
    interactive_approval: bool = False


class ApproveRequest(BaseModel):
    token: str
    decision: bool


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    svc = request.app.state.model_service
    if not await svc.reachable():
        raise HTTPException(
            status_code=503,
            detail="No model backend is reachable. Install mlx-lm (make setup-mlx) "
            "or use the omlx backend.",
        )

    store = request.app.state.sessions
    agent = request.app.state.agent
    session = store.get_or_create(req.session_id, model=req.model)
    approver = (
        InteractiveApprover(request.app.state.pending_approvals)
        if req.interactive_approval
        else None
    )

    async def event_stream():
        # Emit the session id first so the client can continue this conversation.
        yield _sse({"type": "session", "session_id": session.id})
        try:
            # The loop yields fully-formed events (assistant_delta / tool_call /
            # approval_request / tool_result / done / error); forward them verbatim.
            async for event in agent.run(
                session, req.message, req.model, approver=approver
            ):
                yield _sse(event)
            # Persist the completed turn so the conversation survives a restart and shows
            # up in the session list. Off-thread: the write must not stall the response.
            await asyncio.to_thread(store.checkpoint, session)
        except Exception as exc:  # stream the error instead of dropping the connection
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/chat/approve")
async def approve(req: ApproveRequest, request: Request):
    """Resolve a pending interactive approval (Approve/Deny tapped in the GUI)."""
    found = resolve_pending(request.app.state.pending_approvals, req.token, req.decision)
    return {"ok": found}
