"""Turn-scoped unified diff of the files a turn changed.

The agent loop snapshots each write_file/edit_file target's bytes before its first touch
in a turn and reads them again after the turn; this module turns those before/after pairs
into a readable unified diff plus a summary, so a gateway can return "what changed" to the
user (Spring 2 P2/P3 — code results are a first-class result, not just text).

Git-free on purpose: it diffs exactly the files the agent's edit tools touched, so it works
in any workspace and never includes pre-existing unrelated changes.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

# Cap the assembled diff so a huge generated file can't blow up the payload (or a chat
# message). The full files are still on disk; this is only the returned summary view.
_MAX_DIFF_CHARS = 100_000


@dataclass
class FileChange:
    path: str
    status: str  # "added" | "modified" | "deleted"
    additions: int
    deletions: int


@dataclass
class TurnChanges:
    files: list[FileChange] = field(default_factory=list)
    diff: str = ""

    @property
    def total_additions(self) -> int:
        return sum(f.additions for f in self.files)

    @property
    def total_deletions(self) -> int:
        return sum(f.deletions for f in self.files)

    def summary(self) -> str:
        n = len(self.files)
        return (
            f"{n} file{'s' if n != 1 else ''} changed "
            f"(+{self.total_additions}/-{self.total_deletions})"
        )


def _is_binary(b: bytes) -> bool:
    return b"\x00" in b[:8000]


def _status(before: bytes | None, after: bytes | None) -> str:
    if before is None:
        return "added"
    if after is None:
        return "deleted"
    return "modified"


def build_turn_changes(
    snapshots: dict[str, tuple[bytes | None, bytes | None]],
) -> TurnChanges:
    """Build a unified diff from ``path -> (before, after)`` snapshots.

    ``before is None`` means the file was created this turn; ``after is None`` means it was
    removed. Net-unchanged and binary files are recorded in the summary but not diffed.
    """
    changes = TurnChanges()
    parts: list[str] = []
    for path in sorted(snapshots):
        before, after = snapshots[path]
        if before == after:
            continue  # touched but net-unchanged (e.g. write of identical content)
        status = _status(before, after)
        if (before and _is_binary(before)) or (after and _is_binary(after)):
            changes.files.append(FileChange(path, status, 0, 0))
            parts.append(f"# {path} ({status}, binary — not shown)")
            continue
        before_lines = (before or b"").decode("utf-8", "replace").splitlines(keepends=True)
        after_lines = (after or b"").decode("utf-8", "replace").splitlines(keepends=True)
        ud = list(
            difflib.unified_diff(
                before_lines, after_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
            )
        )
        adds = sum(1 for ln in ud if ln.startswith("+") and not ln.startswith("+++"))
        dels = sum(1 for ln in ud if ln.startswith("-") and not ln.startswith("---"))
        changes.files.append(FileChange(path, status, adds, dels))
        parts.append("".join(ud))

    diff = "\n".join(p for p in parts if p)
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n… (diff truncated)"
    changes.diff = diff
    return changes
