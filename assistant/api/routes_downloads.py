"""Model download endpoints.

Downloads a HuggingFace repo into the configured download dir (the same cache mlx-lm / mflux
use, so it's shared and de-duplicated). Discovery's scan then surfaces it in ``/models``. The
lifecycle — progress, cancel, resume-after-restart, retry — lives in
:class:`assistant.downloads.DownloadManager` (``app.state.download_manager``); these routes are
thin wrappers. The client polls ``GET /downloads`` for live progress.
"""

from __future__ import annotations

import importlib.util

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(tags=["downloads"])


class DownloadRequest(BaseModel):
    repo_id: str


def _clean_repo_id(raw: str) -> str:
    repo_id = raw.strip()
    if not repo_id or any(c.isspace() for c in repo_id):
        # A pasted multi-line / doubled value would otherwise reach the hub and fail with a
        # confusing message — reject it up front with a clear one (also guards non-GUI callers).
        raise HTTPException(
            status_code=400,
            detail="repo_id must be a single 'namespace/name' with no spaces or line breaks.",
        )
    return repo_id


def _require_hub() -> None:
    if importlib.util.find_spec("huggingface_hub") is None:
        raise HTTPException(status_code=503, detail="model download requires huggingface_hub.")


@router.post("/models/download")
async def start_download(req: DownloadRequest, request: Request):
    _require_hub()
    repo_id = _clean_repo_id(req.repo_id)
    return request.app.state.download_manager.start(repo_id)


@router.post("/models/download/cancel")
async def cancel_download(req: DownloadRequest, request: Request):
    repo_id = _clean_repo_id(req.repo_id)
    try:
        return request.app.state.download_manager.cancel(repo_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such download: {repo_id}") from None


@router.post("/models/download/retry")
async def retry_download(req: DownloadRequest, request: Request):
    _require_hub()
    repo_id = _clean_repo_id(req.repo_id)
    try:
        return request.app.state.download_manager.retry(repo_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such download: {repo_id}") from None


@router.post("/models/download/remove")
async def remove_download(req: DownloadRequest, request: Request):
    repo_id = _clean_repo_id(req.repo_id)
    try:
        request.app.state.download_manager.remove(repo_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no such download: {repo_id}") from None
    return {"removed": repo_id}


@router.get("/downloads")
async def list_downloads(request: Request):
    return {"downloads": request.app.state.download_manager.snapshot()}
