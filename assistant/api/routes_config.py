"""GUI-editable configuration (a whitelisted subset of Settings).

A small, safe subset is exposed: the model/download paths and the server bind
host/port. A PUT writes the keys into ``$XDG_CONFIG_HOME/assistant/config.toml``
(merging, not clobbering). Discovery-related keys (model dirs, the HF-cache toggle)
are applied to the running backend immediately; only the bind host/port and a backend
swap report ``restart_required`` (repoint-only; existing files are never moved).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from assistant.config import XDG_CONFIG_DIR

router = APIRouter(tags=["config"])

_CONFIG_PATH = XDG_CONFIG_DIR / "config.toml"


_PATH_KEYS = ("models_dir", "download_dir")
_BACKEND_CHOICES = ("mlx", "omlx")


class ConfigPatch(BaseModel):
    models_dir: str | None = None
    download_dir: str | None = None
    extra_model_dirs: list[str] | None = None
    hf_cache: bool | None = None
    backend_host: str | None = None
    backend_port: int | None = None
    model_backend: str | None = None


@router.get("/config")
async def get_config(request: Request):
    s = request.app.state.settings
    return {
        "models_dir": str(s.models_dir),
        "download_dir": str(s.download_dir),
        "extra_model_dirs": [str(p) for p in s.extra_model_dirs],
        "hf_cache": s.hf_cache,
        "backend_host": s.backend_host,
        "backend_port": s.backend_port,
        "model_backend": s.model_backend,
        "config_path": str(_CONFIG_PATH),
    }


@router.put("/config")
async def put_config(patch: ConfigPatch, request: Request):
    updates = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="no editable fields provided")
    for key in _PATH_KEYS:
        if key in updates:
            p = Path(updates[key]).expanduser()
            if not p.is_absolute():
                raise HTTPException(
                    status_code=400, detail=f"{key} must be an absolute path"
                )
            updates[key] = str(p)
    if "extra_model_dirs" in updates:
        resolved: list[str] = []
        for raw in updates["extra_model_dirs"]:
            p = Path(raw).expanduser()
            if not p.is_absolute():
                raise HTTPException(
                    status_code=400,
                    detail="extra_model_dirs entries must be absolute paths",
                )
            resolved.append(str(p))
        updates["extra_model_dirs"] = resolved
    if "backend_port" in updates and not (1 <= updates["backend_port"] <= 65535):
        raise HTTPException(status_code=400, detail="backend_port must be 1–65535")
    if "backend_host" in updates and not updates["backend_host"].strip():
        raise HTTPException(status_code=400, detail="backend_host must not be empty")
    if "model_backend" in updates and updates["model_backend"] not in _BACKEND_CHOICES:
        raise HTTPException(
            status_code=400, detail="model_backend must be 'mlx' or 'omlx'"
        )

    _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if _CONFIG_PATH.is_file():
        existing = tomllib.loads(_CONFIG_PATH.read_text())
    existing.update(updates)
    _CONFIG_PATH.write_text(tomli_w.dumps(existing))

    # Apply discovery-related changes to the LIVE backend so they take effect
    # immediately (discovery is just a filesystem scan). Only host/port and a backend
    # swap genuinely need a process restart — rebinding the socket / reloading the
    # model engine can't be done in place.
    restart_required = _apply_live(request, updates)

    return {
        "updated": updates,
        "restart_required": restart_required,
        "config_path": str(_CONFIG_PATH),
    }


# Keys that take effect live vs. those that can only change on a restart.
_DISCOVERY_KEYS = frozenset({"models_dir", "download_dir", "extra_model_dirs", "hf_cache"})
_RESTART_KEYS = frozenset({"backend_host", "backend_port", "model_backend"})


def _apply_live(request: Request, updates: dict) -> bool:
    """Push discovery changes onto the running settings + model service; return whether
    any change still requires a restart to take effect."""
    settings = request.app.state.settings
    if "models_dir" in updates:
        settings.models_dir = Path(updates["models_dir"])
    if "download_dir" in updates:
        settings.download_dir = Path(updates["download_dir"])
    if "extra_model_dirs" in updates:
        settings.extra_model_dirs = [Path(p) for p in updates["extra_model_dirs"]]
    if "hf_cache" in updates:
        settings.hf_cache = updates["hf_cache"]

    if updates.keys() & _DISCOVERY_KEYS:
        service = getattr(request.app.state, "model_service", None)
        # Only the native MLX service scans local dirs; omlx discovery lives elsewhere.
        if service is not None and hasattr(service, "reconfigure"):
            service.reconfigure(
                models_dir=settings.models_dir,
                extra_model_dirs=settings.extra_model_dirs,
                include_hf_cache=settings.hf_cache,
            )

    return bool(updates.keys() & _RESTART_KEYS)
