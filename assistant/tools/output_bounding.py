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
