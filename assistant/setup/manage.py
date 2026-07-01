"""Preflight checks + managed-tool installation (thin-app / managed-venv model).

The .app ships thin; the Python backend runs from a managed venv. These helpers let
the GUI answer three questions before the user hits a wall: are the data paths there,
which optional MLX tools are installed, and how many models are available — and then
install a missing tool into the *running backend's own venv* on demand.

Install targets pip *package* names (not editable extras) so it works the same in a
dev checkout and a relocated install.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import time
import urllib.request
from collections.abc import Callable
from importlib import metadata
from pathlib import Path

from assistant.config import XDG_CONFIG_DIR, Settings
from assistant.models.mlx_discovery import discover_models

# Managed optional tools, mirroring pyproject's extras. `module` is import-probed;
# `package` is what pip installs.
FEATURES: dict[str, dict[str, str]] = {
    "mlx": {"package": "mlx-lm", "module": "mlx_lm", "label": "LLM inference (mlx-lm)"},
    "images": {"package": "mflux", "module": "mflux", "label": "Image generation (mflux)"},
    "embeddings": {"package": "mlx-embeddings", "module": "mlx_embeddings", "label": "Semantic memory (mlx-embeddings)"},
    "vlm": {"package": "mlx-vlm", "module": "mlx_vlm", "label": "Read images (mlx-vlm)"},
    "audio": {"package": "mlx-audio", "module": "mlx_audio", "label": "Speech STT / TTS (mlx-audio)"},
    "video": {"package": "mlx-video", "module": "mlx_video", "label": "Video generation (mlx-video)"},
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        # A half-installed package can raise rather than return None.
        return False


def _installed_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except Exception:
        # PackageNotFoundError (not installed) or any metadata oddity → unknown.
        return None


def _is_newer(latest: str | None, installed: str | None) -> bool:
    """True only when we can prove PyPI's latest is strictly newer than what's installed.
    Unknown on either side → False, so we never nag with an update we can't justify."""
    if not latest or not installed:
        return False
    try:
        from packaging.version import Version

        return Version(latest) > Version(installed)
    except Exception:
        # No packaging, or unparseable version (a git build): fall back to inequality.
        return latest != installed


# PyPI "latest version" lookups are cached: preflight is polled every few seconds, but a
# package release cadence is days — re-checking the network each poll would be wasteful and
# fragile. Keep the last value for 6h; on any network failure keep the previous answer.
_PYPI_TTL = 6 * 3600.0
_pypi_cache: dict[str, tuple[str | None, float]] = {}


def _pypi_latest(package: str, *, now: float) -> str | None:
    cached = _pypi_cache.get(package)
    if cached is not None and now - cached[1] < _PYPI_TTL:
        return cached[0]
    latest = cached[0] if cached else None  # default: keep last known on failure
    try:
        req = urllib.request.Request(
            f"https://pypi.org/pypi/{package}/json",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            latest = json.load(resp).get("info", {}).get("version") or latest
    except Exception:
        pass  # offline / 404 / timeout → keep last known (or None); never raise
    _pypi_cache[package] = (latest, now)
    return latest


def _queryable_tools(settings: Settings) -> list[tuple[str, str]]:
    """(package, module) for tools whose PyPI latest is worth checking: installed and not
    source-overridden. Shared by the networked fetch and the non-networked cache reads so
    all three agree on exactly which tools participate."""
    sources = settings.managed_tool_sources or {}
    return [
        (meta["package"], meta["module"])
        for key, meta in FEATURES.items()
        if key not in sources and _installed(meta["module"])
    ]


def fetch_latest_versions(settings: Settings) -> dict[str, str | None]:
    """PyPI latest for installed, non-source tools (cached). Networked — call off the
    event loop. Source-overridden tools (e.g. a patched mlx-lm git build) are skipped:
    they update by re-pulling the source, so PyPI's version is irrelevant to them."""
    now = time.time()
    return {
        package: _pypi_latest(package, now=now)
        for package, _module in _queryable_tools(settings)
    }


def cached_latest_versions(settings: Settings) -> dict[str, str | None]:
    """Last-known PyPI-latest versions from the in-process cache, WITHOUT touching the
    network. Mirrors ``fetch_latest_versions``' tool selection but returns ``None`` for any
    tool not yet fetched. This lets ``/preflight`` answer instantly on a freshly spawned
    backend (cold cache) instead of blocking on up to 2s/pkg of sequential PyPI lookups —
    a real refresh warms the cache in the background and later polls pick it up. An absent
    value means "no update info yet", which the GUI already treats as "no update"."""
    out: dict[str, str | None] = {}
    for package, _module in _queryable_tools(settings):
        cached = _pypi_cache.get(package)
        out[package] = cached[0] if cached else None
    return out


def latest_versions_fresh(settings: Settings, *, now: float | None = None) -> bool:
    """True when every queryable tool has a within-TTL cache entry — i.e. a background
    refresh would be a no-op. Lets the caller avoid scheduling redundant network work on
    every poll (``/preflight`` is polled every few seconds)."""
    now = time.time() if now is None else now
    for package, _module in _queryable_tools(settings):
        cached = _pypi_cache.get(package)
        if cached is None or now - cached[1] >= _PYPI_TTL:
            return False
    return True


def check_tools(
    settings: Settings | None = None, *, latest: dict[str, str | None] | None = None
) -> list[dict]:
    """Managed-tool status. ``latest`` (PyPI versions, from ``fetch_latest_versions``)
    drives ``update_available`` for PyPI tools; source-overridden tools always offer a
    re-pull when present. Called with no args it degrades to install-state only (no
    update detection) — used by tests and any caller that doesn't need versions."""
    sources = (settings.managed_tool_sources if settings else None) or {}
    latest = latest or {}
    tools = []
    for key, meta in FEATURES.items():
        package = meta["package"]
        installed = _installed(meta["module"])
        version = _installed_version(package) if installed else None
        source = sources.get(key)
        if source:
            # A patched/git source has no PyPI version to compare — offer a refresh
            # whenever the tool is present so the user can re-pull the latest build (N5/N11).
            update = installed
        else:
            update = bool(installed and _is_newer(latest.get(package), version))
        tools.append(
            {
                "feature": key,
                "package": package,
                "label": meta["label"],
                "installed": installed,
                "version": version,
                "latest": latest.get(package),
                "source": source,
                "update_available": update,
            }
        )
    return tools


def check_paths(settings: Settings) -> list[dict]:
    entries = {
        "data": settings.home_dir,
        "models": settings.models_dir,
        "skills": settings.skills_dir,
        "memory": settings.memory_dir,
        "images": settings.images_dir,
        "audio": settings.audio_dir,
        "video": settings.video_dir,
    }
    return [
        {"name": name, "path": str(p), "exists": Path(p).is_dir()}
        for name, p in entries.items()
    ]


def model_summary(settings: Settings) -> dict:
    try:
        count = len(
            discover_models(
                settings.models_dir,
                include_hf_cache=settings.hf_cache,
                extra_dirs=settings.extra_model_dirs,
            )
        )
    except Exception:
        count = 0
    return {
        "dir": str(settings.models_dir),
        "exists": Path(settings.models_dir).is_dir(),
        "count": count,
        "hf_cache": settings.hf_cache,
    }


def preflight(
    settings: Settings, *, latest: dict[str, str | None] | None = None
) -> dict:
    config_path = XDG_CONFIG_DIR / "config.toml"
    return {
        "venv": sys.prefix,
        "python": sys.version.split()[0],
        "config_path": str(config_path),
        "config_exists": config_path.is_file(),
        "download_dir": str(settings.download_dir),
        "paths": check_paths(settings),
        "tools": check_tools(settings, latest=latest),
        "models": model_summary(settings),
    }


def find_uv() -> str | None:
    """Locate the uv binary without trusting the inherited PATH.

    The SwiftUI app spawns this backend with macOS's minimal GUI PATH (``/usr/bin:/bin``
    …), which omits user tool dirs (``~/.local/bin``, a mise shim dir, Homebrew). A bare
    ``shutil.which('uv')`` therefore misses an installed uv, and we wrongly fall back to
    the venv's (absent) pip — the cause of the "No module named pip" install failure. An
    ``ASSISTANT_UV``/``UV`` override wins so the app can hand us the exact uv it used.
    """
    override = os.environ.get("ASSISTANT_UV") or os.environ.get("UV")
    if override and Path(override).is_file():
        return override
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local/bin/uv",
        Path.home() / ".local/share/mise/shims/uv",
        Path.home() / ".cargo/bin/uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def install_command(
    feature: str, *, uv: str | None, upgrade: bool = False, source: str | None = None
) -> list[str]:
    """The command that installs `feature` into THIS backend's venv.

    With uv we install via ``uv pip`` (fast, and what the bootstrap used). Without it we
    fall back to the venv's own pip — which the install runner bootstraps via
    ``ensurepip`` first, because uv-created venvs ship without pip.

    ``source`` overrides the install target with a pip spec (e.g. a patched mlx-lm git
    build that supports newer model architectures than the PyPI wheel — N11). For a source
    upgrade we force a clean reinstall (the ref is often a moving branch/PR a plain
    ``--upgrade`` would no-op); for a plain PyPI upgrade ``--upgrade`` climbs to latest.
    """
    target = source or FEATURES[feature]["package"]
    if upgrade and source:
        flags = ["--reinstall"] if uv else ["--force-reinstall"]
    elif upgrade:
        flags = ["--upgrade"]
    else:
        flags = []
    if uv:
        return [uv, "pip", "install", "--python", sys.executable, *flags, target]
    return [sys.executable, "-m", "pip", "install", *flags, target]


async def perform_install(
    state: dict, feature: str, runner: Callable[[str], object]
) -> None:
    """Run one install, recording its lifecycle in `state` (testable: the runner is
    injected so this never shells out under test)."""
    package = FEATURES[feature]["package"]
    state[feature] = {"feature": feature, "package": package, "status": "installing", "error": None}
    try:
        await asyncio.to_thread(runner, feature)
        state[feature] = {"feature": feature, "package": package, "status": "done", "error": None}
    except Exception as exc:
        state[feature] = {"feature": feature, "package": package, "status": "error", "error": str(exc)}
