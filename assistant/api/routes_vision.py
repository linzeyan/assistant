from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["vision"])


class VisionRequest(BaseModel):
    path: str
    question: str | None = None


@router.post("/vision/describe")
async def describe_image(req: VisionRequest, request: Request):
    svc = request.app.state.vision
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="vision unavailable; install mlx-vlm on Apple Silicon.",
        )
    try:
        text = await svc.describe(
            [req.path], req.question or "Describe this image in detail."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"vision failed: {exc}") from exc
    return {"text": text}
