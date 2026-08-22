#!/usr/bin/env python3
"""Count how often each FIND block in a brief occurs in the file the brief names.

    ./check-anchors.py brief.md [workspace]

Exits non-zero when any anchor does not appear exactly once, which is the cheap
way to catch the brief's most common defect before spending a turn on it.

Each anchor is counted twice: exactly, and with leading whitespace stripped from
every line. The second count is the one that matters. The model routinely
re-sends an anchor with its indentation dropped, so a line that is unique *with*
its indentation (`        checkFoo()`) can still match the declaration it was
named after (`    private static func checkFoo() {`) — the edit tool then refuses
it as ambiguous and the model retries the identical call until the driver kills
it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Which file the anchors that follow belong to. Either form the brief template
# uses: a heading whose text ends in a backticked path, or the "Read only" line
# a single-file brief opens with.
FILE_HEADING = re.compile(r"^#+ .*`([^`]+)`\s*$", re.M)
READ_ONLY = re.compile(r"^Read only `([^`]+)`", re.M)
# A fenced FIND block in any language, or none.
FIND_BLOCK = re.compile(r"^FIND[^\n]*:\n```[a-zA-Z0-9_+-]*\n(.*?)\n```", re.S | re.M)


def loose(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines())


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    brief = Path(sys.argv[1])
    root = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    text = brief.read_text(encoding="utf-8")

    current: str | None = None
    anchors: list[tuple[str, str, str]] = []
    # Split on headings so an anchor is attributed to the file heading above it.
    for chunk in re.split(r"\n(?=#+ )", text):
        for pattern in (FILE_HEADING, READ_ONLY):
            found = pattern.search(chunk)
            # A heading ending in backticks is a file heading only if what it
            # quotes looks like a path — otherwise `### Edit 2 — rename `foo()``
            # would silently redirect every anchor under it at a file called
            # `foo()`, and the reported counts would be about nothing.
            if found and ("/" in found.group(1) or "." in found.group(1)):
                current = found.group(1)
                break
        for block in FIND_BLOCK.finditer(chunk):
            if current is None:
                sys.exit(f"{brief}: a FIND block appears before any file is named")
            anchors.append((current, chunk.splitlines()[0].strip(), block.group(1)))

    if not anchors:
        sys.exit(f"{brief}: no FIND blocks found — is this a brief?")

    bad = 0
    for path, title, find in anchors:
        body = (root / path).read_text(encoding="utf-8")
        exact, relaxed = body.count(find), loose(body).count(loose(find))
        problem = not (exact == 1 and relaxed == 1)
        bad += problem
        flag = "  <-- PROBLEM" if problem else ""
        print(f"{path}  exact={exact} loose={relaxed}  {title}{flag}")
    print("OK" if not bad else f"{bad} problem(s): fix the brief before dispatching it")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
