"""Shell execution tool.

Runs in its own process session so a timeout can kill the whole process group
(not just the shell), and is approval-gated since arbitrary commands are dangerous.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from .base import ToolContext, ToolResult
from .output_bounding import bound_text
from .registry import registry

log = logging.getLogger("assistant")

_MAX_OUTPUT = 60_000

# Resolved once per process by _user_path(); None means "resolution failed, use what we
# inherited". The sentinel distinguishes that from "not tried yet".
_UNRESOLVED = object()
_user_path_cache: object = _UNRESOLVED
_user_path_lock = asyncio.Lock()


async def _user_path() -> str | None:
    """PATH as the user's own login shell reports it.

    Launched from Assistant.app, the backend inherits launchd's PATH —
    ``/usr/bin:/bin:/usr/sbin:/sbin`` — which has no Homebrew, no cargo, no rg, and no
    version-manager shims. Every one of those is a tool the user has installed and expects
    this agent to be able to run, and `command not found` reads to a model as "not
    installed": observed across a long coding session, an agent asked to verify its work
    with `cargo test` got exit 127 every time, concluded the toolchain was missing, and
    reported the task done without ever having built it.

    Asking the user's own shell is the only way to get the PATH they actually have — it is
    assembled by their profile, not by anything this process can see. Resolved once and
    reused; on failure we keep the inherited PATH rather than inventing one.
    """
    global _user_path_cache
    if _user_path_cache is not _UNRESOLVED:
        return _user_path_cache  # type: ignore[return-value]
    async with _user_path_lock:
        if _user_path_cache is not _UNRESOLVED:
            return _user_path_cache  # type: ignore[return-value]
        shell = os.environ.get("SHELL") or "/bin/sh"
        resolved: str | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                shell, "-lic", 'printf %s "$PATH"',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
            )
            # An interactive login shell runs the user's whole profile, which can block on a
            # prompt or a slow plugin. Bounded, and a timeout is just a failed resolution.
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
            candidate = out.decode(errors="replace").strip()
            if candidate and os.pathsep in candidate:
                resolved = candidate
        except (OSError, asyncio.TimeoutError, TimeoutError):
            resolved = None
        if resolved is None:
            log.warning("could not resolve the login shell's PATH; using the inherited one")
        else:
            log.info("resolved login shell PATH (%d entries)", resolved.count(os.pathsep) + 1)
        _user_path_cache = resolved
        return resolved


@registry.tool(
    name="bash",
    description="Run a shell command and return its combined stdout+stderr and exit code.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {
                "type": "string",
                "description": "Working directory (defaults to the workspace).",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 120).",
            },
        },
        "required": ["command"],
    },
    needs_approval=True,
)
async def bash(args: dict, ctx: ToolContext) -> ToolResult:
    command = args["command"]
    cwd = args.get("cwd") or str(ctx.cwd)
    timeout = float(args.get("timeout", 120))

    path = await _user_path()
    env = {**os.environ, "PATH": path} if path else None

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group -> clean kill on timeout
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, TimeoutError):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return ToolResult(False, f"timed out after {timeout:g}s")

    text = out.decode(errors="replace")
    # Bound combined stdout+stderr keeping both ends; the command's real result/error is
    # usually at the tail. Ephemeral output, so spill the full text for on-demand retrieval.
    text = bound_text(
        text, limit=_MAX_OUTPUT, spill_dir=ctx.output_spill_dir, label="bash"
    )
    return ToolResult(proc.returncode == 0, f"[exit {proc.returncode}]\n{text}")
