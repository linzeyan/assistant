"""Discover shell-touched file changes for the turn diff, via git.

write/edit tools snapshot their own targets (loop._snapshot_before_edit); the bash tool can
touch anything, so we lean on git to find what changed during the turn. Bounded and
turn-scoped: on the first bash call we record the set of already-dirty files and snapshot
their bytes (their true "before"); at turn end we look at what's dirty now and feed each
changed file's before/after into the same diff builder. A file that was clean before bash uses
its committed content as "before" (correct — clean means working == HEAD); a new untracked
file has no "before" (it's an addition).

git is optional: outside a repo this captures nothing and the write/edit diff still works.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_TIMEOUT = 10


def _git(cwd, *args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=text,
        timeout=_TIMEOUT,
    )


def repo_root(cwd) -> Path | None:
    try:
        r = _git(cwd, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return None
    out = r.stdout.strip() if r.returncode == 0 else ""
    return Path(out) if out else None


def dirty_paths(root: Path) -> dict[str, str]:
    """Map repo-relative path -> porcelain status for every changed/untracked file."""
    try:
        r = _git(root, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {}
    if r.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in r.stdout.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:]
        if " -> " in path:  # rename: "R  old -> new" — the new path is what exists now
            path = path.split(" -> ", 1)[1]
        out[path.strip('"')] = code
    return out


def head_bytes(root: Path, relpath: str) -> bytes | None:
    """The committed (HEAD) bytes of a path, or None if it isn't tracked there."""
    try:
        r = _git(root, "show", f"HEAD:{relpath}", text=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None
