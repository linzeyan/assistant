"""Connect-or-spawn lifecycle for the omlx server.

omlx is distributed via Homebrew (NOT PyPI), so we cannot ``pip install`` or import
it. We instead either attach to an already-running instance (brew services or the
omlx.app) or spawn ``omlx serve`` as a managed child process. Spawning uses a new
session (own process group) so ``stop()`` can tear down the whole tree cleanly —
mirroring omlx's own SignalHandlers approach.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from pathlib import Path

from .omlx_client import OmlxClient
from .status import BackendState, BackendStatus

# Back-compat aliases: this module historically owned these names, and the rest of
# the file (plus tests) still refers to them. The canonical types now live in
# ``status`` so the native MLX backend can share them.
OmlxState = BackendState
OmlxStatus = BackendStatus


# GUI-launched processes often inherit a minimal PATH that lacks /opt/homebrew/bin,
# so probe the canonical Homebrew locations explicitly as a fallback.
_HOMEBREW_BINS = ("/opt/homebrew/bin/omlx", "/usr/local/bin/omlx")


class OmlxProcess:
    def __init__(
        self,
        client: OmlxClient,
        *,
        host: str,
        port: int,
        models_dir: Path | None,
        omlx_bin: str | None,
        autostart: bool,
        startup_timeout: float = 40.0,
    ):
        self._client = client
        self._host = host
        self._port = port
        self._models_dir = models_dir
        self._omlx_bin = omlx_bin
        self._autostart = autostart
        self._startup_timeout = startup_timeout
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _resolve_bin(self) -> str | None:
        if self._omlx_bin:
            return self._omlx_bin if os.path.exists(self._omlx_bin) else None
        found = shutil.which("omlx")
        if found:
            return found
        for cand in _HOMEBREW_BINS:
            if os.path.exists(cand):
                return cand
        return None

    async def ensure_running(self) -> OmlxStatus:
        if await self._client.health():
            return OmlxStatus(
                OmlxState.CONNECTED, "Attached to a running omlx server.", self.base_url
            )
        if not self._autostart:
            return OmlxStatus(
                OmlxState.UNAVAILABLE,
                "omlx not reachable and autostart is disabled.",
                self.base_url,
            )
        bin_path = self._resolve_bin()
        if not bin_path:
            return OmlxStatus(
                OmlxState.UNAVAILABLE,
                "omlx not found. Install it with `brew install omlx`.",
                self.base_url,
            )
        await self._spawn(bin_path)
        if await self._await_health():
            return OmlxStatus(
                OmlxState.SPAWNED, f"Started omlx ({bin_path}).", self.base_url
            )
        await self.stop()
        return OmlxStatus(
            OmlxState.UNAVAILABLE,
            "omlx was started but did not become healthy within the timeout.",
            self.base_url,
        )

    async def _spawn(self, bin_path: str) -> None:
        args = [bin_path, "serve", "--host", self._host, "--port", str(self._port)]
        if self._models_dir:
            args += ["--model-dir", str(self._models_dir)]
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,  # own process group → clean group kill on stop
        )

    async def _await_health(self) -> bool:
        waited, step = 0.0, 0.5
        while waited < self._startup_timeout:
            if self._proc and self._proc.returncode is not None:
                return False  # the child exited early — give up
            if await self._client.health():
                return True
            await asyncio.sleep(step)
            waited += step
        return False

    async def stop(self) -> None:
        """Terminate the spawned omlx (no-op if we merely attached to a running one)."""
        proc = self._proc
        if proc is None or proc.returncode is not None:
            self._proc = None
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            self._proc = None
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (asyncio.TimeoutError, TimeoutError):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._proc = None
