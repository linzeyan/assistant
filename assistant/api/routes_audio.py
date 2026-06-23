from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["audio"])


class TranscribeRequest(BaseModel):
    path: str


class SpeakRequest(BaseModel):
    text: str


@router.post("/audio/transcribe")
async def transcribe(req: TranscribeRequest, request: Request):
    svc = request.app.state.audio
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="audio unavailable; install mlx-audio on Apple Silicon.",
        )
    try:
        text = await svc.transcribe(req.path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
    return {"text": text}


@router.post("/audio/speech")
async def speech(req: SpeakRequest, request: Request):
    svc = request.app.state.audio
    if svc is None or not svc.available():
        raise HTTPException(
            status_code=503,
            detail="audio unavailable; install mlx-audio on Apple Silicon.",
        )
    try:
        path = await svc.speak(req.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc
    return {"path": str(path)}
