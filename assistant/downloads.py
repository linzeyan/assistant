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
import contextlib
import functools
import importlib.util
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("assistant")

# Rate/ETA are recomputed on this cadence as the AVERAGE over the trailing window (not a jittery
# per-second EMA), so the displayed speed and ETA are stable.
_RATE_WINDOW_S = 10.0
# Below this average rate the transfer is effectively stalled — reporting an ETA then is worse than
# useless: remaining/rate explodes to an astronomical integer that (a) is meaningless and (b) once
# overflowed Int64, breaking the GUI's JSON decode of the whole downloads list. So we report ETA
# "unknown" (None) when the rate is this low OR the ETA would exceed the sane ceiling below.
_ETA_MIN_RATE_BPS = 1024.0  # 1 KB/s
_ETA_MAX_S = 30 * 24 * 3600  # 30 days; a longer estimate is noise — report unknown instead

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
    # hf_transfer is the Rust chunk-parallel downloader: it opens several connections PER FILE, so it
    # saturates the link even at max_workers=1. A single plain-HTTPS stream to the HF CDN was measured
    # capping ~1.9 MB/s (and stalling), while the huggingface-cli — which uses hf_transfer — hit
    # 5-8 MB/s at the same worker count. Enable it whenever the package is importable; hf_hub raises
    # if the flag is set without it, so gate on the import. (With Xet on, Xet wins and this is inert.)
    if importlib.util.find_spec("hf_transfer") is not None:
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
        # Downloaded is clamped to the repo total: progress polls the directory size, which also
        # counts the hub's .cache/huggingface bookkeeping — stale *.incomplete files from an
        # earlier cancelled attempt made the GUI show "54.24 GB / 53.81 GB · 100%" while still
        # transferring. The clamp keeps the pair honest; the real bytes land at "done".
        downloaded = (
            min(self.downloaded_bytes, self.total_bytes)
            if self.total_bytes
            else self.downloaded_bytes
        )
        # ETA is bounded on BOTH ends so it can never blow up: a stalled/near-zero rate or an
        # absurdly long estimate is reported as unknown (None) rather than a giant integer that
        # overflows the client's Int64 and corrupts the whole downloads response. Zero remaining
        # (clamped progress, tail still verifying/moving) is unknown too — "ETA 0s" is a lie.
        eta = None
        if self.status == "downloading" and self.rate_bps >= _ETA_MIN_RATE_BPS and self.total_bytes:
            remaining = max(0, self.total_bytes - downloaded)
            secs = int(remaining / self.rate_bps)
            eta = secs if 0 < secs <= _ETA_MAX_S else None
        return {
            "repo_id": self.repo_id,
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": downloaded,
            "eta_seconds": eta,
            # Current transfer speed (10s-window average) so the GUI can show "· 12.3 MB/s". Only
            # meaningful while downloading; None otherwise.
            "rate_bps": self.rate_bps if self.status == "downloading" else None,
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
        # Single-active gate (N52): model files are large, so exactly ONE download transfers at a
        # time — the rest wait their turn in "queued". Constructed here (no running loop needed on
        # 3.10+); it binds to the loop on first acquire.
        self._gate = asyncio.Semaphore(1)
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
            cancel.set()  # a downloading item: the runner kills its subprocess immediately
        self._cancel_queued_task(repo_id, state)
        return state.to_public()

    def remove(self, repo_id: str) -> None:
        """Drop a download from the list entirely so a finished/cancelled/failed entry doesn't
        linger forever. If it's still in flight, cancel it first (kills the subprocess); the
        orphaned task then unwinds against a detached state and won't re-add the entry."""
        if repo_id not in self._states:
            raise KeyError(repo_id)
        state = self._states[repo_id]
        cancel = self._cancels.get(repo_id)
        if cancel is not None:
            cancel.set()
        self._cancel_queued_task(repo_id, state)
        self._states.pop(repo_id, None)
        self._persist()

    def _cancel_queued_task(self, repo_id: str, state: DownloadState) -> None:
        # A still-queued download is blocked on the gate; the cancel event won't wake it (no
        # subprocess exists yet), so cancel its task — it unwinds to "cancelled" without waiting for
        # its turn. Safe: status flips to "downloading" synchronously before any subprocess starts,
        # so a "queued" status guarantees there's nothing to orphan.
        if state.status == "queued":
            task = self._tasks.get(repo_id)
            if task is not None:
                task.cancel()

    async def resume_incomplete(self) -> None:
        """Re-spawn downloads that were mid-flight when the app last closed (the hub continues
        their partial files). Called once at startup."""
        self._reap_orphan_downloads()  # kill leftovers from a previous backend before re-spawning
        for repo_id, state in list(self._states.items()):
            if state.status in ACTIVE_STATUSES:
                log.info("resuming interrupted download: %s", repo_id)
                self._spawn(repo_id)

    def _reap_orphan_downloads(self) -> None:
        """Kill download subprocesses left over from a PREVIOUS backend instance. A download runs in
        a detached session (start_new_session) so it SURVIVES a backend restart — and resume would
        then spawn a DUPLICATE. Two processes downloading the same repo contend on the hub's per-file
        .lock and the transfer crawls, then deadlocks at 0 B/s (observed: an orphan with PPID=1 next
        to the resumed one). A true orphan is reparented to launchd (PPID=1) — a subprocess whose
        parent is still alive belongs to ANOTHER RUNNING backend (observed: a manual restart racing
        the app's supervisor made the two backends reap each other's in-flight downloads). Matched
        narrowly — our exact `python -c` body AND a path under our target dir — so a user's
        unrelated `huggingface-cli download` is never touched. Best-effort: pgrep/pkill may be absent."""
        with contextlib.suppress(Exception):
            found = subprocess.run(
                ["pgrep", "-f", "huggingface_hub import snapshot_download"],
                capture_output=True, text=True,
            )
            for pid in found.stdout.split():
                cmd = subprocess.run(
                    ["ps", "-p", pid, "-o", "command="], capture_output=True, text=True
                ).stdout
                if str(self._target_dir) not in cmd:
                    continue  # someone else's download, not ours — leave it alone
                ppid = subprocess.run(
                    ["ps", "-p", pid, "-o", "ppid="], capture_output=True, text=True
                ).stdout.strip()
                if ppid != "1":
                    continue  # parent still alive -> another backend owns it — leave it alone
                with contextlib.suppress(ProcessLookupError, ValueError, PermissionError):
                    os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
                    log.warning("reaped orphaned download subprocess pid=%s", pid)

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
        state.error = None
        state.status = "queued"  # normalized so resumed/retried entries also wait at the gate
        cancel = asyncio.Event()
        self._cancels[repo_id] = cancel
        self._persist()
        try:
            # Wait our turn: only one download holds the gate at a time. A still-queued download is
            # blocked HERE, so cancel()/remove() cancels its task to unblock it without waiting.
            async with self._gate:
                if cancel.is_set():  # cancelled/removed while queued
                    state.status = "cancelled"
                    return
                # From here on a subprocess may exist; flip to "downloading" (and persist) BEFORE
                # the runner starts so cancel() sees "downloading" and kills the subprocess via the
                # event rather than cancelling the task and orphaning it.
                state.status = "downloading"
                self._persist()
                try:
                    state.total_bytes = await asyncio.to_thread(self._size_fn, repo_id)
                except Exception as exc:  # sizing is best-effort; bar falls back to bytes-only
                    log.warning("could not size %s: %s", repo_id, exc)
                    state.total_bytes = 0
                await self._runner(repo_id, self._target_dir / repo_id, state, cancel)
                if cancel.is_set():
                    state.status = "cancelled"
                else:
                    state.status = "done"
                    if state.total_bytes:
                        state.downloaded_bytes = state.total_bytes
                    log.info("download done: %s (%d bytes)", repo_id, state.downloaded_bytes)
        except asyncio.CancelledError:
            state.status = "cancelled"
            raise
        except Exception as exc:
            state.status = "cancelled" if cancel.is_set() else "error"
            if state.status == "error":
                state.error = str(exc)
                log.error("download failed: %s: %s", repo_id, exc)
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
    log.info(
        "download start: %s -> %s (max_workers=%d, xet_disabled=%s)",
        repo_id, target, max_workers, str((env or {}).get("HF_HUB_DISABLE_XET") == "1"),
    )
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
        # transfer (e.g. HF_HUB_DISABLE_XET, HF_HUB_DOWNLOAD_TIMEOUT). Progress bars are disabled
        # because we render our own from disk size — and an undrained tqdm stream can fill the
        # stderr pipe and BACK-PRESSURE (stall) the child. Caller env still wins if it overrides.
        env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", **(env or {})},
    )

    async def _watch_cancel() -> None:
        await cancel.wait()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)  # interrupt mid-file
        except ProcessLookupError:
            pass

    # Drain stderr CONTINUOUSLY into a bounded tail. Before, stderr was read only on exit, so a
    # chatty child could fill the OS pipe buffer and block mid-download (invisible stall) — and any
    # retry/timeout diagnostics were lost. Now the pipe never fills, and HF's warnings are logged.
    stderr_tail: deque[str] = deque(maxlen=40)

    async def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            raw = await proc.stderr.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            stderr_tail.append(line)
            low = line.lower()
            if any(k in low for k in ("error", "retry", "timeout", "failed", "warning")):
                log.warning("download %s: %s", repo_id, line)

    watcher = asyncio.create_task(_watch_cancel())
    drainer = asyncio.create_task(_drain_stderr())
    # Rate/ETA are a trailing-window AVERAGE refreshed every _RATE_WINDOW_S; progress bytes still
    # update every second so the bar stays smooth and cancel stays responsive.
    rate_anchor_t, rate_anchor_b = time.monotonic(), state.downloaded_bytes
    try:
        while True:
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
                break  # process exited
            except (asyncio.TimeoutError, TimeoutError):
                now = time.monotonic()
                bytes_now = _dir_size(target)
                state.downloaded_bytes = bytes_now
                window = now - rate_anchor_t
                if window >= _RATE_WINDOW_S:
                    state.rate_bps = max(0.0, (bytes_now - rate_anchor_b) / window)
                    rate_anchor_t, rate_anchor_b = now, bytes_now
        if cancel.is_set():
            return
        await drainer  # let the last stderr lines land before we inspect the tail
        if proc.returncode != 0:
            # Lead with the exit code: the stderr tail alone can be pure noise (a harmless
            # FutureWarning masked a SIGKILL — the surfaced "error" said deprecation, not death).
            tail = "\n".join(stderr_tail)[-500:].strip()
            raise RuntimeError(f"exit {proc.returncode}" + (f": {tail}" if tail else ""))
        state.downloaded_bytes = _dir_size(target) or state.downloaded_bytes
    finally:
        watcher.cancel()
        drainer.cancel()
