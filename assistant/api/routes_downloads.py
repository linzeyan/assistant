"""Model download endpoints.

Downloads a HuggingFace repo into the local HF cache via ``snapshot_download`` (the
same cache mlx-lm / mflux use, so it's shared and de-duplicated). Discovery's HF
cache scan then surfaces it in ``/models``. Downloads run as background tasks; the
client polls ``GET /downloads`` for status.
"""

from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["downloads"])


class DownloadRequest(BaseModel):
    repo_id: str


async def perform_download(
    state: dict, repo_id: str, downloader: Callable[[str], object]
) -> None:
    """Run one download, recording its lifecycle in ``state`` (testable: the
    downloader is injected so this never needs the network)."""
    state[repo_id] = {"repo_id": repo_id, "status": "downloading", "error": None}
    try:
        await asyncio.to_thread(downloader, repo_id)
        state[repo_id] = {"repo_id": repo_id, "status": "done", "error": None}
    except Exception as exc:
        state[repo_id] = {"repo_id": repo_id, "status": "error", "error": str(exc)}


@router.post("/models/download")
async def start_download(req: DownloadRequest, request: Request):
    if importlib.util.find_spec("huggingface_hub") is None:
        raise HTTPException(
            status_code=503, detail="model download requires huggingface_hub."
        )
    state = request.app.state.downloads
    if state.get(req.repo_id, {}).get("status") == "downloading":
        return {"repo_id": req.repo_id, "status": "downloading"}  # idempotent

    from huggingface_hub import snapshot_download

    # Fetch into the configured download dir (defaults to models_dir) as org/model,
    # which discovery's two-level scan then surfaces in /models.
    target = Path(request.app.state.settings.download_dir)

    def downloader(repo_id: str) -> object:
        return snapshot_download(repo_id, local_dir=str(target / repo_id))

    task = asyncio.create_task(perform_download(state, req.repo_id, downloader))
    request.app.state.download_tasks[req.repo_id] = task
    # Drop the finished task reference so the registry doesn't grow unbounded.
    task.add_done_callback(
        lambda _t, rid=req.repo_id: request.app.state.download_tasks.pop(rid, None)
    )
    return {"repo_id": req.repo_id, "status": "downloading"}


@router.get("/downloads")
async def list_downloads(request: Request):
    return {"downloads": list(request.app.state.downloads.values())}
