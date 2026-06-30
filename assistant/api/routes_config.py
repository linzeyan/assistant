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
from assistant.gateway import lifecycle as gateway_lifecycle

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
    max_output_tokens: int | None = None
    max_tool_iters: int | None = None
    # Gateways (S9): a token/allowlist edit (re)starts the gateway live, no backend restart.
    # An empty token clears it (stops the gateway).
    telegram_token: str | None = None
    telegram_allowed_users: list[int] | None = None


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
        "max_output_tokens": s.max_output_tokens,
        "max_tool_iters": s.max_tool_iters,
        "config_path": str(_CONFIG_PATH),
        **gateway_lifecycle.status(request.app),  # telegram_* (token masked)
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
    if "max_output_tokens" in updates and not (64 <= updates["max_output_tokens"] <= 131072):
        raise HTTPException(
            status_code=400, detail="max_output_tokens must be 64–131072"
        )
    if "max_tool_iters" in updates and not (1 <= updates["max_tool_iters"] <= 100):
        raise HTTPException(status_code=400, detail="max_tool_iters must be 1–100")
    if "telegram_token" in updates:
        token = updates["telegram_token"].strip()
        if token and any(c.isspace() for c in token):
            raise HTTPException(
                status_code=400, detail="telegram_token must not contain whitespace."
            )
        updates["telegram_token"] = token  # normalise; "" clears the gateway

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

    # Gateways (S9): a token/allowlist change (re)starts the gateway live — no backend restart.
    if updates.keys() & _GATEWAY_KEYS:
        await gateway_lifecycle.reload(request.app)

    return {
        "updated": updates,
        "restart_required": restart_required,
        "config_path": str(_CONFIG_PATH),
    }


# Keys that take effect live vs. those that can only change on a restart.
_DISCOVERY_KEYS = frozenset({"models_dir", "download_dir", "extra_model_dirs", "hf_cache"})
_RESTART_KEYS = frozenset({"backend_host", "backend_port", "model_backend"})
_GATEWAY_KEYS = frozenset({"telegram_token", "telegram_allowed_users"})


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
    if "max_output_tokens" in updates:
        # Applies live: the next turn reads the loop's ceiling, so no restart needed.
        settings.max_output_tokens = updates["max_output_tokens"]
        agent = getattr(request.app.state, "agent", None)
        if agent is not None:
            agent.set_max_output_tokens(updates["max_output_tokens"])
    if "max_tool_iters" in updates:
        # Applies live: the next turn's loop reads the new budget, so no restart needed.
        settings.max_tool_iters = updates["max_tool_iters"]
        agent = getattr(request.app.state, "agent", None)
        if agent is not None:
            agent.set_max_iters(updates["max_tool_iters"])
    # Gateway settings: stage onto live settings so the subsequent reload reads the new values
    # (the reload itself is async, so it runs in put_config, not here).
    if "telegram_token" in updates:
        settings.telegram_token = updates["telegram_token"] or None
    if "telegram_allowed_users" in updates:
        settings.telegram_allowed_users = list(updates["telegram_allowed_users"])

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
