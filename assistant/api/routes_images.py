from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["images"])


class ImageRequest(BaseModel):
    prompt: str
    steps: int | None = None
    seed: int | None = None


class EditImageRequest(BaseModel):
    prompt: str
    # Accept either a single path or a list (multi-reference edit); at least one required.
    image_paths: list[str] | None = None
    image_path: str | None = None
    steps: int | None = None
    seed: int | None = None


@router.post("/images/generate")
async def generate_image(req: ImageRequest, request: Request):
    svc = request.app.state.images
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="image generation unavailable; install mflux on Apple Silicon.",
        )
    try:
        path = await svc.generate_image(req.prompt, steps=req.steps, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return {"path": str(path)}


@router.post("/images/edit")
async def edit_image(req: EditImageRequest, request: Request):
    svc = request.app.state.images
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="image editing unavailable; install mflux>=0.18 on Apple Silicon.",
        )
    paths = req.image_paths or ([req.image_path] if req.image_path else [])
    if not paths:
        raise HTTPException(
            status_code=422, detail="image_paths or image_path is required."
        )
    try:
        path = await svc.edit_image(req.prompt, paths, steps=req.steps, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"edit failed: {exc}") from exc
    return {"path": str(path)}
