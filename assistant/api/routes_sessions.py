"""Conversation session CRUD (spring1 S1).

The GUI lists past conversations, opens one to resume it, starts new ones, and deletes
them. Sessions are persisted by ``SessionStore`` (file-per-session JSON), so this router
is a thin shell over it. The chat turn itself still flows through ``POST /chat``, which
checkpoints the session after each turn.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    model: str | None = None


@router.get("/sessions")
async def list_sessions(request: Request):
    return {"sessions": request.app.state.sessions.list_sessions()}


@router.post("/sessions")
async def create_session(req: CreateSessionRequest, request: Request):
    session = request.app.state.sessions.create(model=req.model)
    return session.summary()


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return {
        "id": session.id,
        "model": session.model,
        "title": session.title or session.derived_title(),
        "messages": session.messages,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    ok = request.app.state.sessions.delete_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    return {"ok": True}
