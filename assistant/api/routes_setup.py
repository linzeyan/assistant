"""Preflight + managed-tool install endpoints.

`GET /preflight` reports what the GUI needs to guide first-run setup (paths, tools,
model count). `POST /setup/install` installs a missing tool into the backend's own
venv as a background task; the client polls via `/preflight` (which folds in install
state) or `GET /setup/installs`.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from functools import partial

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from assistant.setup.manage import (
    FEATURES,
    fetch_latest_versions,
    find_uv,
    install_command,
    perform_install,
    preflight,
)

router = APIRouter(tags=["setup"])


class InstallRequest(BaseModel):
    feature: str
    upgrade: bool = False


def _run_install(
    feature: str, *, upgrade: bool = False, source: str | None = None
) -> None:
    """Default runner: shell out to uv (preferred) or the venv's pip. Raises on a
    non-zero exit so the lifecycle helper records an error. ``source`` (a configured
    install spec, e.g. a patched mlx-lm git build) overrides the PyPI package target."""
    uv = find_uv()
    if uv is None:
        # uv-created venvs ship without pip, and the GUI-spawned backend may not see a
        # uv on its minimal PATH — bootstrap pip into the venv so the fallback below can
        # run. Best effort: any failure here surfaces as the pip command's own error.
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
            text=True,
        )
    proc = subprocess.run(
        install_command(feature, uv=uv, upgrade=upgrade, source=source),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise RuntimeError(tail or f"install exited with {proc.returncode}")


@router.get("/preflight")
async def get_preflight(request: Request):
    settings = request.app.state.settings
    # PyPI "latest" lookups touch the network — run them off the event loop (cached, so
    # only the rare cache-miss actually does I/O). Drives the version-gated update button.
    latest = await asyncio.to_thread(fetch_latest_versions, settings)
    report = preflight(settings, latest=latest)
    # Surface in-flight / finished installs so the GUI can show progress inline.
    report["installs"] = list(request.app.state.installs.values())
    return report


@router.post("/setup/install")
async def start_install(req: InstallRequest, request: Request):
    if req.feature not in FEATURES:
        raise HTTPException(status_code=404, detail=f"unknown feature: {req.feature}")
    state = request.app.state.installs
    if state.get(req.feature, {}).get("status") == "installing":
        return {"feature": req.feature, "status": "installing"}  # idempotent

    settings = request.app.state.settings
    source = (settings.managed_tool_sources or {}).get(req.feature)
    task = asyncio.create_task(
        perform_install(
            state,
            req.feature,
            partial(_run_install, upgrade=req.upgrade, source=source),
        )
    )
    request.app.state.install_tasks[req.feature] = task
    task.add_done_callback(
        lambda _t, f=req.feature: request.app.state.install_tasks.pop(f, None)
    )
    return {"feature": req.feature, "status": "installing"}


@router.get("/setup/installs")
async def list_installs(request: Request):
    return {"installs": list(request.app.state.installs.values())}
