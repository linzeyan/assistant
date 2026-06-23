from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["memory"])


class MemoryCreate(BaseModel):
    content: str
    tags: list[str] = []


class MemoryUpdate(BaseModel):
    content: str
    tags: list[str] = []


@router.get("/memory")
async def list_memory(request: Request):
    return {"memories": await request.app.state.memory.all()}


@router.get("/memory/search")
async def search_memory(request: Request, q: str, limit: int = 5):
    return {"results": await request.app.state.memory.search(q, limit=limit)}


@router.post("/memory")
async def create_memory(payload: MemoryCreate, request: Request):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    return await request.app.state.memory.write(payload.content, payload.tags)


@router.put("/memory/{entry_id}")
async def update_memory(entry_id: str, payload: MemoryUpdate, request: Request):
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="content is required")
    updated = await request.app.state.memory.update(
        entry_id, payload.content, payload.tags
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"no memory with id '{entry_id}'")
    return updated


@router.delete("/memory/{entry_id}")
async def delete_memory(entry_id: str, request: Request):
    if not await request.app.state.memory.delete(entry_id):
        raise HTTPException(status_code=404, detail=f"no memory with id '{entry_id}'")
    return {"ok": True}
