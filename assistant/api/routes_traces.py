"""Per-turn trace read API (spring2 P0) + maintenance wipe.

Lets you scan a session's turns by outcome and open one to see exactly where it died
(model text → parsed calls → tool results). This is the *measure* step before fixing
reliability — there is no write path; turns are recorded by the agent loop as a side
effect of running. The only mutation is the maintenance wipe (DELETE /traces).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["traces"])


def _store(request: Request):
    store = getattr(request.app.state, "trace_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="tracing is disabled")
    return store


@router.get("/sessions/{session_id}/turns")
async def list_turns(session_id: str, request: Request):
    """Turn summaries for a session, newest first — scan the ``outcome`` column for
    ``parse_miss`` / ``tool_error`` to find the failures fast."""
    return {"turns": _store(request).list_for_session(session_id)}


@router.get("/turns/{turn_id}")
async def get_turn(turn_id: str, request: Request):
    trace = _store(request).get(turn_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"unknown turn: {turn_id}")
    return trace


@router.delete("/traces")
async def clear_traces(request: Request):
    """Wipe every recorded turn trace, memory and disk (Settings ▸ Maintenance ▸ "Clear
    traces"). Traces accumulate one JSON file per turn; long dogfood sessions bloat the dir
    and old traces have no value once their bug is fixed."""
    return {"cleared": _store(request).clear()}
