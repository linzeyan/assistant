"""File-oriented coding tools: read / write / edit / glob / grep.

Implemented in pure Python (no rg/fd dependency) so the assistant works on any
end-user machine. write_file and edit_file are approval-gated because they mutate
the filesystem; the read-only tools are not.
"""

from __future__ import annotations

import re
from pathlib import Path

from .base import ToolContext, ToolResult
from .output_bounding import bound_text
from .registry import registry

_MAX_READ = 100_000
_MAX_MATCHES = 200


def _resolve(path: str, ctx: ToolContext) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ctx.cwd / p)


@registry.tool(
    name="read_file",
    description="Read a UTF-8 text file and return its contents.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)
async def read_file(args: dict, ctx: ToolContext) -> ToolResult:
    p = _resolve(args["path"], ctx)
    if not p.is_file():
        return ToolResult(False, f"not a file: {p}")
    data = p.read_text(errors="replace")
    # Keep both ends of a large file (head-only dropped the tail). No spill: the file itself
    # is the full copy on disk — the marker reports the true size so the agent knows.
    return ToolResult(True, bound_text(data, limit=_MAX_READ, label="read"))


@registry.tool(
    name="write_file",
    description="Create or overwrite a text file with the given content.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    needs_approval=True,
)
async def write_file(args: dict, ctx: ToolContext) -> ToolResult:
    p = _resolve(args["path"], ctx)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(args["content"])
    return ToolResult(True, f"wrote {len(args['content'])} chars to {p}")


@registry.tool(
    name="edit_file",
    description="Replace a unique snippet of text in a file with new text.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Existing text; must be unique."},
            "new": {"type": "string"},
        },
        "required": ["path", "old", "new"],
    },
    needs_approval=True,
)
async def edit_file(args: dict, ctx: ToolContext) -> ToolResult:
    p = _resolve(args["path"], ctx)
    if not p.is_file():
        return ToolResult(False, f"not a file: {p}")
    text = p.read_text(errors="replace")
    old = args["old"]
    count = text.count(old)
    if count == 0:
        return ToolResult(False, "old snippet not found")
    if count > 1:
        return ToolResult(False, f"old snippet is not unique ({count} occurrences)")
    p.write_text(text.replace(old, args["new"], 1))
    return ToolResult(True, f"edited {p}")


@registry.tool(
    name="glob",
    description="List files matching a glob pattern under the workspace.",
    parameters={
        "type": "object",
        "properties": {"pattern": {"type": "string"}},
        "required": ["pattern"],
    },
)
async def glob_files(args: dict, ctx: ToolContext) -> ToolResult:
    matches = sorted(str(x) for x in ctx.cwd.glob(args["pattern"]))
    if not matches:
        return ToolResult(True, "(no matches)")
    return ToolResult(True, "\n".join(matches[:_MAX_MATCHES]))


def _is_file(p: Path) -> bool:
    """``is_file()`` raises rather than returning False on a path this process can't stat at all
    (macOS system paths like /usr/sbin/weakpass_edit, other users' directories). Unguarded, a
    single such entry anywhere under the root aborted the whole search — the caller got an error
    instead of the matches from every readable file."""
    try:
        return p.is_file()
    except OSError:
        return False


@registry.tool(
    name="grep",
    description="Search file contents for a regular expression.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {
                "type": "string",
                "description": "File or directory to search (defaults to the workspace).",
            },
        },
        "required": ["pattern"],
    },
)
async def grep(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        rx = re.compile(args["pattern"])
    except re.error as exc:
        return ToolResult(False, f"bad regex: {exc}")
    root = _resolve(args["path"], ctx) if args.get("path") else ctx.cwd
    files = [root] if _is_file(root) else (p for p in root.rglob("*") if _is_file(p))
    hits: list[str] = []
    for f in files:
        try:
            lines = f.read_text(errors="strict").splitlines()
        except (UnicodeDecodeError, OSError):
            continue  # skip binary / unreadable files
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                hits.append(f"{f}:{i}: {line.strip()[:200]}")
                if len(hits) >= _MAX_MATCHES:
                    hits.append("...[truncated]")
                    return ToolResult(True, "\n".join(hits))
    return ToolResult(True, "\n".join(hits) if hits else "(no matches)")
