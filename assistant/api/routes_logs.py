"""Log maintenance: clear the backend log files (Settings → "clear logs").

The backend writes to ``<log_dir>/backend.log`` (rotated to ``.1/.2/.3``) and the GUI tees the
spawn's console to ``backend.out.log``. A long download-debugging session bloats these, so the
Settings screen offers a one-tap truncate.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(tags=["logs"])
log = logging.getLogger("assistant")

# The backend's rotating log + its backups, plus the GUI's console tee. Truncated in place (the
# RotatingFileHandler keeps appending afterwards), best-effort per file.
_LOG_FILES = ("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.out.log")


@router.post("/logs/clear")
async def clear_logs(request: Request):
    """Truncate the backend log files. Best-effort per file so a locked/rotated one doesn't fail
    the whole call; returns the names actually cleared."""
    log_dir: Path = request.app.state.settings.log_dir
    cleared: list[str] = []
    for name in _LOG_FILES:
        path = log_dir / name
        if path.exists():
            with contextlib.suppress(OSError):
                path.write_text("")  # truncate in place
                cleared.append(name)
    log.info("cleared logs: %s", ", ".join(cleared) or "(none)")
    return {"cleared": cleared}
