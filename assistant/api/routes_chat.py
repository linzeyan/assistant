from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from assistant.api.sse import with_keepalive
from assistant.tools.approval import InteractiveApprover, resolve_pending

router = APIRouter(tags=["chat"])
log = logging.getLogger("assistant")


class ChatRequest(BaseModel):
    message: str
    model: str
    session_id: str | None = None
    # When set, approval-required tools pause and stream an ``approval_request``
    # event; the client decides via POST /chat/approve. The desktop GUI sets this.
    interactive_approval: bool = False
    # Optional per-request tool-iteration ceiling (H7); overrides the global default for this turn
    # only. Clamped server-side (see below). None / non-positive → the configured default.
    max_iters: int | None = None
    # Working directory for this turn's file/shell tools, overriding the configured
    # workspace_dir. The loop has always taken a per-run cwd — Telegram's /cd rides it — but
    # over HTTP the whole backend shared one directory, so a client driving several isolated
    # checkouts (git worktrees, one task per tree) had no way to keep them apart. None keeps
    # the server default.
    workspace: str | None = None


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

    # Rejected here rather than at first tool call: a bad path would otherwise surface as a
    # confusing "not a file" several tool calls into a turn the client has already paid for.
    workspace = None
    if req.workspace is not None:
        workspace = Path(req.workspace).expanduser()
        if not workspace.is_dir():
            raise HTTPException(status_code=400, detail=f"not a directory: {req.workspace}")
        workspace = str(workspace.resolve())

    store = request.app.state.sessions
    agent = request.app.state.agent
    session = store.get_or_create(req.session_id, model=req.model)
    new_session = req.session_id is None
    log.info(
        "chat turn: session=%s (%s) model=%s msgs=%d",
        session.id, "new" if new_session else "resume", req.model, len(session.messages),
    )
    approver = (
        InteractiveApprover(request.app.state.pending_approvals)
        if req.interactive_approval
        else None
    )

    async def event_stream():
        # Emit the session id first so the client can continue this conversation.
        yield _sse({"type": "session", "session_id": session.id})
        completed = False
        try:
            # The loop yields fully-formed events (assistant_delta / tool_call /
            # approval_request / tool_result / done / error); forward them verbatim.
            # Clamp the per-request override to a sane band so a caller can't request a runaway
            # loop (or a zero that would disable the backstop); None passes through as "default".
            req_max_iters = min(req.max_iters, 100) if req.max_iters else None
            async for event in agent.run(
                session,
                req.message,
                req.model,
                approver=approver,
                cwd=workspace,
                max_iters=req_max_iters,
            ):
                yield _sse(event)
            completed = True
            # Persist the completed turn so the conversation survives a restart and shows
            # up in the session list. Off-thread: the write must not stall the response.
            await asyncio.to_thread(store.checkpoint, session)
            log.info("chat turn checkpointed: session=%s msgs=%d", session.id, len(session.messages))
        except asyncio.CancelledError:
            # The client disconnected mid-stream (New chat / Stop while generating, or the
            # SSE connection dropped). Logged because a turn cancelled here never reaches the
            # checkpoint above — a prime suspect for "conversation vanished after New" (#5).
            log.warning(
                "chat turn cancelled mid-stream: session=%s completed=%s", session.id, completed
            )
            raise
        except Exception as exc:  # stream the error instead of dropping the connection
            log.exception("chat turn failed: session=%s", session.id)
            yield _sse({"type": "error", "detail": str(exc)})
            # Persist even a failed turn so the conversation — and the user's message — doesn't
            # vanish from the session list (#4). The turn added the user message before any
            # exception, so there's real content to keep. Best-effort: a checkpoint failure
            # here must not re-break an already-failed turn.
            try:
                await asyncio.to_thread(store.checkpoint, session)
                log.info(
                    "failed turn checkpointed: session=%s msgs=%d",
                    session.id, len(session.messages),
                )
            except Exception:
                log.exception("checkpoint after failed turn failed: session=%s", session.id)

    # SSE-comment keepalive: silence during queueing/prefill/suppressed tool-call
    # buffering must not look like a dead connection to client read-timeouts (N81).
    # Every consumer parses only ``data:`` lines, so the comment is invisible.
    return StreamingResponse(
        with_keepalive(event_stream()), media_type="text/event-stream"
    )


@router.post("/chat/approve")
async def approve(req: ApproveRequest, request: Request):
    """Resolve a pending interactive approval (Approve/Deny tapped in the GUI)."""
    found = resolve_pending(request.app.state.pending_approvals, req.token, req.decision)
    return {"ok": found}
