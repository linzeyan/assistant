from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from assistant.skills.discovery import normalize_slug
from assistant.skills.frontmatter import parse_frontmatter
from assistant.skills.manage import create_skill, delete_skill

router = APIRouter(tags=["skills"])


def _is_user_skill(store, meta) -> bool:
    """User-authored skills (under the user dir) are editable; bundled ones are read-only."""
    try:
        meta.directory.relative_to(store.user_dir)
        return True
    except ValueError:
        return False


class SkillCreate(BaseModel):
    name: str
    description: str = ""
    body: str


class SkillUpdate(BaseModel):
    description: str = ""
    body: str


class SkillImport(BaseModel):
    content: str  # a full SKILL.md (frontmatter + body)
    overwrite: bool = False


@router.get("/skills")
async def list_skills(request: Request):
    store = request.app.state.skills
    return {
        "skills": [
            {
                "name": m.name,
                "description": m.description,
                "editable": _is_user_skill(store, m),
            }
            for m in sorted(store.index().values(), key=lambda m: m.name)
        ]
    }


@router.post("/skills/reload")
async def reload_skills(request: Request):
    return request.app.state.skills.reload()


@router.post("/skills")
async def create_skill_route(payload: SkillCreate, request: Request):
    if not payload.name.strip() or not payload.body.strip():
        raise HTTPException(status_code=400, detail="name and body are required")
    store = request.app.state.skills
    ok, msg = create_skill(store, payload.name, payload.description, payload.body)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "name": normalize_slug(payload.name), "message": msg}


@router.post("/skills/import")
async def import_skill(payload: SkillImport, request: Request):
    meta, body = parse_frontmatter(payload.content)
    name = str(meta.get("name") or "").strip()
    if not name:
        raise HTTPException(
            status_code=400, detail="imported file needs a 'name' in its frontmatter"
        )
    if not body.strip():
        raise HTTPException(status_code=400, detail="imported file has no body")
    description = str(meta.get("description") or "")
    store = request.app.state.skills
    ok, msg = create_skill(store, name, description, body, overwrite=payload.overwrite)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "name": normalize_slug(name), "message": msg}


@router.get("/skills/{name}")
async def view_skill(name: str, request: Request):
    store = request.app.state.skills
    meta = store.get(name)
    body = store.read_body(name)
    if meta is None or body is None:
        raise HTTPException(status_code=404, detail=f"unknown skill: {name}")
    return {
        "name": meta.name,
        "description": meta.description,
        "body": body,
        "editable": _is_user_skill(store, meta),
    }


@router.put("/skills/{name}")
async def update_skill(name: str, payload: SkillUpdate, request: Request):
    if not payload.body.strip():
        raise HTTPException(status_code=400, detail="body is required")
    store = request.app.state.skills
    meta = store.get(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown skill: {name}")
    # Copy-on-write: editing a bundled skill writes a user-dir copy that shadows it
    # (scan order makes user skills win), so it's customisable without ever mutating the
    # read-only bundle. A later delete archives the copy and the original re-emerges.
    ok, msg = create_skill(
        store, meta.name, payload.description, payload.body, overwrite=True
    )
    if not ok:  # an update of an existing skill shouldn't collide, but stay defensive
        raise HTTPException(status_code=409, detail=msg)
    return {"ok": True, "name": meta.name, "message": msg}


@router.delete("/skills/{name}")
async def delete_skill_route(name: str, request: Request):
    store = request.app.state.skills
    meta = store.get(name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown skill: {name}")
    if not _is_user_skill(store, meta):
        raise HTTPException(
            status_code=403,
            detail=f"'{meta.name}' is a bundled skill and cannot be deleted",
        )
    ok, msg = delete_skill(store, name)  # archives, never hard-deletes
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "name": meta.name, "message": msg}
