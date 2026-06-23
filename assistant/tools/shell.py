"""Shell execution tool.

Runs in its own process session so a timeout can kill the whole process group
(not just the shell), and is approval-gated since arbitrary commands are dangerous.
"""

from __future__ import annotations

import asyncio
import os
import signal

from .base import ToolContext, ToolResult
from .registry import registry

_MAX_OUTPUT = 60_000


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

    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=cwd,
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
    if len(text) > _MAX_OUTPUT:
        text = text[:_MAX_OUTPUT] + "\n...[truncated]"
    return ToolResult(proc.returncode == 0, f"[exit {proc.returncode}]\n{text}")
