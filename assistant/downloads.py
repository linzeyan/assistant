"""Model download manager (N17): progress, cancel, resume-after-restart, retry.

``snapshot_download`` in a worker thread can't be interrupted, so each download runs as a
**subprocess we can kill** — cancellation is immediate, even mid-large-file (the user's hard
requirement). Progress is derived from the repo's total size (``HfApi``) versus the bytes on
disk (polled), so there's no fragile stdout parsing. In-flight downloads are persisted to
``downloads.json`` and resumed on the next startup; failed/cancelled ones can be retried — the
hub continues partial files either way.

The size lookup and the runner are injectable so the lifecycle is testable without the network
or a real subprocess.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import signal
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("assistant")

# Subprocess body: download one repo into a local dir. Args arrive via argv (no shell, so a
# crafted repo_id can't inject); runs in the SAME interpreter as the backend, sidestepping the
# GUI-spawn PATH gap where `huggingface-cli` isn't resolvable.
_DOWNLOAD_SCRIPT = (
    "import sys; from huggingface_hub import snapshot_download; "
    "snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=int(sys.argv[3]))"
)

# Statuses for which a download is still live (idempotent start, resume-on-boot).
ACTIVE_STATUSES = frozenset({"queued", "downloading"})


def hub_env(*, disable_xet: bool, download_timeout: int) -> dict[str, str]:
    """The extra hub env for a download subprocess, from the user-tunable settings. Single source
    so main.py (startup) and the live config PUT build it identically. HF_HUB_DISABLE_XET is the
    big one — Xet was measured throttling to a few KB/s on some networks."""
    env = {"HF_HUB_DOWNLOAD_TIMEOUT": str(download_timeout)}
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    return env


@dataclass
class DownloadState:
    repo_id: str
    status: str = "queued"  # queued | downloading | done | error | cancelled
    total_bytes: int = 0
    downloaded_bytes: int = 0
    error: str | None = None
    rate_bps: float = 0.0  # runtime-only EMA; never persisted

    def to_public(self) -> dict:
        eta = None
        if self.status == "downloading" and self.rate_bps > 0 and self.total_bytes:
            remaining = max(0, self.total_bytes - self.downloaded_bytes)
            eta = int(remaining / self.rate_bps)
        return {
            "repo_id": self.repo_id,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "eta_seconds": eta,
            "error": self.error,
        }

    def to_persisted(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "error": self.error,
        }


SizeFn = Callable[[str], int]
Runner = Callable[[str, Path, DownloadState, asyncio.Event], Awaitable[None]]


class DownloadManager:
    def __init__(
        self,
        *,
        target_dir: Path,
        state_path: Path,
        size_fn: SizeFn | None = None,
        runner: Runner | None = None,
        env: dict[str, str] | None = None,
        max_workers: int = 8,
    ):
        self._target_dir = Path(target_dir)
        self._state_path = Path(state_path)
        self._size_fn = size_fn or _hf_total_size
        # Bind the tunables (extra env like HF_HUB_DISABLE_XET, and snapshot_download's max_workers)
        # onto the default subprocess runner, so the call site stays the fixed Runner signature and
        # an injected test runner is unaffected.
        self._runner = runner or functools.partial(
            _subprocess_runner, env=env, max_workers=max_workers
        )
        self._states: dict[str, DownloadState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._cancels: dict[str, asyncio.Event] = {}
        self._load()

    # --- public API ---
    def snapshot(self) -> list[dict]:
        return [s.to_public() for s in self._states.values()]

    def set_download_options(self, *, env: dict[str, str] | None, max_workers: int) -> None:
        """Update the download tunables live (from the config PUT). Rebinds the default subprocess
        runner; only downloads STARTED AFTER this call see the change — an in-flight subprocess keeps
        the env/workers it was spawned with. No-op safety isn't needed: routes_config only calls this
        on the real manager (tests inject their own runner and never call it)."""
        self._runner = functools.partial(
            _subprocess_runner, env=env, max_workers=max_workers
        )

    def start(self, repo_id: str) -> dict:
        existing = self._states.get(repo_id)
        if existing and existing.status in ACTIVE_STATUSES:
            return existing.to_public()  # idempotent: don't double-spawn
        self._states[repo_id] = DownloadState(repo_id=repo_id, status="queued")
        self._spawn(repo_id)
        return self._states[repo_id].to_public()

    def retry(self, repo_id: str) -> dict:
        state = self._states[repo_id]  # KeyError -> 404 at the route
        if state.status in ACTIVE_STATUSES:
            return state.to_public()
        state.status = "queued"
        state.error = None
        self._spawn(repo_id)
        return state.to_public()

    def cancel(self, repo_id: str) -> dict:
        state = self._states[repo_id]  # KeyError -> 404 at the route
        cancel = self._cancels.get(repo_id)
        if cancel is not None:
            cancel.set()  # the runner kills the subprocess immediately
        return state.to_public()

    def remove(self, repo_id: str) -> None:
        """Drop a download from the list entirely so a finished/cancelled/failed entry doesn't
        linger forever. If it's still in flight, cancel it first (kills the subprocess); the
        orphaned task then unwinds against a detached state and won't re-add the entry."""
        if repo_id not in self._states:
            raise KeyError(repo_id)
        cancel = self._cancels.get(repo_id)
        if cancel is not None:
            cancel.set()
        self._states.pop(repo_id, None)
        self._persist()

    async def resume_incomplete(self) -> None:
        """Re-spawn downloads that were mid-flight when the app last closed (the hub continues
        their partial files). Called once at startup."""
        for repo_id, state in list(self._states.items()):
            if state.status in ACTIVE_STATUSES:
                log.info("resuming interrupted download: %s", repo_id)
                self._spawn(repo_id)

    async def aclose(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()

    # --- internals ---
    def _spawn(self, repo_id: str) -> None:
        task = asyncio.create_task(self._download_one(repo_id))
        self._tasks[repo_id] = task
        task.add_done_callback(lambda _t, rid=repo_id: self._tasks.pop(rid, None))

    async def _download_one(self, repo_id: str) -> None:
        state = self._states[repo_id]
        state.status = "downloading"
        state.error = None
        cancel = asyncio.Event()
        self._cancels[repo_id] = cancel
        self._persist()
        try:
            state.total_bytes = await asyncio.to_thread(self._size_fn, repo_id)
        except Exception as exc:  # sizing is best-effort; bar falls back to bytes-only
            log.warning("could not size %s: %s", repo_id, exc)
            state.total_bytes = 0
        try:
            await self._runner(repo_id, self._target_dir / repo_id, state, cancel)
            if cancel.is_set():
                state.status = "cancelled"
            else:
                state.status = "done"
                if state.total_bytes:
                    state.downloaded_bytes = state.total_bytes
        except asyncio.CancelledError:
            state.status = "cancelled"
            raise
        except Exception as exc:
            state.status = "cancelled" if cancel.is_set() else "error"
            if state.status == "error":
                state.error = str(exc)
        finally:
            state.rate_bps = 0.0
            self._cancels.pop(repo_id, None)
            self._persist()

    def _persist(self) -> None:
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"downloads": [s.to_persisted() for s in self._states.values()]}
            tmp = self._state_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2))
            os.replace(tmp, self._state_path)  # atomic
        except OSError as exc:
            log.warning("could not persist downloads.json: %s", exc)

    def _load(self) -> None:
        if not self._state_path.is_file():
            return
        try:
            data = json.loads(self._state_path.read_text())
        except (OSError, ValueError) as exc:
            log.warning("could not read downloads.json: %s", exc)
            return
        for entry in data.get("downloads", []):
            rid = entry.get("repo_id")
            if not rid:
                continue
            self._states[rid] = DownloadState(
                repo_id=rid,
                status=entry.get("status", "error"),
                total_bytes=entry.get("total_bytes", 0),
                downloaded_bytes=entry.get("downloaded_bytes", 0),
                error=entry.get("error"),
            )


# --- real (network/subprocess) implementations; injected away in tests ----------------------

def _hf_total_size(repo_id: str) -> int:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    return sum((sib.size or 0) for sib in (info.siblings or []))


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass  # file vanished mid-walk (rename/cleanup) — skip
    return total


async def _subprocess_runner(
    repo_id: str,
    target: Path,
    state: DownloadState,
    cancel: asyncio.Event,
    *,
    env: dict[str, str] | None = None,
    max_workers: int = 8,
) -> None:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _DOWNLOAD_SCRIPT,
        repo_id,
        str(target),
        str(max_workers),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # own group -> kill the whole tree on cancel
        # Merge onto (not replace) the parent env so PATH/HOME survive; extra keys tune the hub
        # transfer (e.g. HF_HUB_DISABLE_XET, HF_HUB_DOWNLOAD_TIMEOUT).
        env={**os.environ, **(env or {})},
    )

    async def _watch_cancel() -> None:
        await cancel.wait()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # interrupt mid-file
        except ProcessLookupError:
            pass

    watcher = asyncio.create_task(_watch_cancel())
    last_t, last_b = time.monotonic(), state.downloaded_bytes
    try:
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break  # process exited
            except (asyncio.TimeoutError, TimeoutError):
                now = time.monotonic()
                bytes_now = _dir_size(target)
                dt = now - last_t
                if dt > 0:
                    inst = max(0.0, (bytes_now - last_b) / dt)
                    state.rate_bps = inst if state.rate_bps == 0 else 0.7 * state.rate_bps + 0.3 * inst
                state.downloaded_bytes = bytes_now
                last_t, last_b = now, bytes_now
        if cancel.is_set():
            return
        if proc.returncode != 0:
            err = await proc.stderr.read() if proc.stderr is not None else b""
            raise RuntimeError(err.decode(errors="replace")[-500:].strip() or f"exit {proc.returncode}")
        state.downloaded_bytes = _dir_size(target) or state.downloaded_bytes
    finally:
        watcher.cancel()
