"""Unified tool-output bounding (spring1 S4).

A tool's output flows straight into the model's context, so it must be bounded. The old
per-tool caps were single-sided (`text[:N]`) — they kept the HEAD and dropped the END, which
for shell output, logs, and tracebacks is exactly where the result or error lives. This keeps
BOTH ends (head + tail) with a clear marker for the elided middle, and — when a spill dir is
given — writes the full output to a file and names it in the marker, so the agent can read the
rest on demand instead of losing it.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

# Default budget per tool result (chars). The model's context is the scarce resource, not
# disk — tools pass their own limit; this is the fallback.
DEFAULT_LIMIT = 60_000


def bound_text(
    text: str,
    *,
    limit: int = DEFAULT_LIMIT,
    spill_dir: Path | None = None,
    label: str = "output",
) -> str:
    """Bound ``text`` to ~``limit`` chars, keeping both ends. Returns it unchanged when already
    within budget. When over budget and ``spill_dir`` is given, the full text is written there
    (content-addressed) and its path is named in the marker for on-demand retrieval."""
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    # Keep both ends; the tail (errors/results/summaries) is as important as the head.
    head_len = limit // 2
    tail_len = limit - head_len
    head, tail = text[:head_len], text[-tail_len:]
    spill_note = ""
    if spill_dir is not None:
        path = _spill(text, spill_dir, label)
        if path is not None:
            spill_note = f"; full {len(text)} chars at {path}"
    marker = f"\n\n...[{omitted} chars omitted{spill_note}]...\n\n"
    return head + marker + tail


def gc_spill_dir(spill_dir: Path, *, max_age_days: float) -> tuple[int, int]:
    """Prune spilled tool-output files older than ``max_age_days`` (S15 spill GC).

    Spill files are content-addressed and otherwise never deleted, so the dir grows without bound.
    A spill pointer is consumed shortly after it's written (same turn/session), so age-based pruning
    is safe: a stale pointer in a long-dormant conversation just degrades to "file gone" and the tool
    can be re-run. ``max_age_days <= 0`` disables GC (keep forever). Best-effort — a stat/unlink error
    on one file never aborts the sweep. Returns ``(files_removed, bytes_freed)``."""
    if max_age_days <= 0 or not spill_dir.is_dir():
        return (0, 0)
    cutoff = time.time() - max_age_days * 86_400
    removed = freed = 0
    for path in spill_dir.glob("*.txt"):
        try:
            st = path.stat()
            if st.st_mtime < cutoff:
                path.unlink()
                removed += 1
                freed += st.st_size
        except OSError:
            continue  # racing unlink / permission — skip, keep sweeping
    return (removed, freed)


def _spill(text: str, spill_dir: Path, label: str) -> Path | None:
    """Write the full text to ``spill_dir`` and return its path. Content-addressed so identical
    output reuses one file and the name is stable (the agent can re-read it). Best-effort: a
    write failure returns None and bounding proceeds without a spill pointer."""
    try:
        spill_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]
        path = spill_dir / f"{label}-{digest}.txt"
        if not path.exists():
            path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        return None
