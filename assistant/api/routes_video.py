from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["video"])


class VideoRequest(BaseModel):
    prompt: str
    resolution: str | None = None  # "360p" (default) / "480p" / "540p" / "720p"
    num_frames: int | None = None
    steps: int | None = None
    seed: int | None = None
    negative_prompt: str | None = None


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
            req.prompt,
            resolution=req.resolution,
            num_frames=req.num_frames,
            steps=req.steps,
            seed=req.seed,
            negative_prompt=req.negative_prompt,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc
    return {"path": str(path)}
