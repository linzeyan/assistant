"""Conversation session CRUD (spring1 S1).

The GUI lists past conversations, opens one to resume it, starts new ones, and deletes
them. Sessions are persisted by ``SessionStore`` (file-per-session JSON), so this router
is a thin shell over it. The chat turn itself still flows through ``POST /chat``, which
checkpoints the session after each turn.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from assistant.agent.tokens import estimate_messages_tokens

router = APIRouter(tags=["sessions"])
log = logging.getLogger("assistant")


class CreateSessionRequest(BaseModel):
    model: str | None = None


@router.get("/sessions")
async def list_sessions(request: Request):
    sessions = request.app.state.sessions.list_sessions()
    # Logged so "my old conversation vanished" (#5) is diagnosable: this is exactly how many
    # the backend can see (in-memory + on-disk merged) at the moment the GUI asks.
    log.info("list sessions: %d found", len(sessions))
    return {"sessions": sessions}


@router.post("/sessions")
async def create_session(req: CreateSessionRequest, request: Request):
    session = request.app.state.sessions.create(model=req.model)
    return session.summary()


@router.get("/sessions/search")
async def search_sessions(request: Request, q: str = "", limit: int = 20):
    """Cross-session full-text search (F/S14). Declared BEFORE ``/sessions/{session_id}`` so the
    literal ``search`` path isn't captured as a session id by the dynamic route."""
    results = request.app.state.sessions.search_sessions(q, limit=max(1, min(limit, 100)))
    return {"query": q, "count": len(results), "results": results}


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


@router.post("/sessions/{session_id}/compact")
async def compact_session(session_id: str, request: Request):
    """Manually compact a session now (S6), independent of the auto threshold. Summarizes the
    oldest turns, keeps recent ones verbatim, and archives the originals on the session."""
    store = request.app.state.sessions
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {session_id}")
    compaction = getattr(request.app.state, "compaction", None)
    if compaction is None:
        raise HTTPException(status_code=503, detail="compaction is disabled")
    if not session.model:
        raise HTTPException(
            status_code=400, detail="session has no model to summarize with"
        )
    event = await compaction.force_compact(session, session.model)
    if event is None:
        # Nothing old enough to summarize safely — report current size, no change made.
        return {
            "compacted": False,
            "context_tokens": estimate_messages_tokens(session.messages),
        }
    await asyncio.to_thread(store.checkpoint, session)
    return {"compacted": True, **event}
