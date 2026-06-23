from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["video"])


class VideoRequest(BaseModel):
    prompt: str
    num_frames: int | None = None
    seed: int | None = None


@router.post("/video/generate")
async def generate_video(req: VideoRequest, request: Request):
    svc = request.app.state.video
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="video generation unavailable; install mlx-video on Apple Silicon.",
        )
    try:
        path = await svc.generate_video(
            req.prompt, num_frames=req.num_frames, seed=req.seed
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return {"path": str(path)}
