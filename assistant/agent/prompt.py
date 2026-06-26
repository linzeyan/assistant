"""System-prompt assembly: base instructions + skills index.

Only the skills INDEX (name + description) is injected, never every skill body —
that keeps the prompt small and lets the model pull a skill's full text on demand
via skill_view.

The system prompt is deliberately STABLE (it depends only on the base text + the skills
index, not on the user's message), so it can serve as a byte-identical cacheable prefix
across turns (on-device KV-cache reuse). Per-turn memory does NOT go here — it rides the
current user turn via ``wrap_memory_context`` so dynamic context can't perturb the prefix.
"""

from __future__ import annotations

from datetime import datetime

_BASE = """You are a local-first AI coding assistant running on the user's Mac.

You have tools to read/write/edit files, run shell commands, search code, search the \
web and fetch pages, manage skills, and store/recall long-term memory. Prefer using \
tools over guessing.

Current / external information:
- The current date is provided with each user message — use it; do not guess the date \
from your training data.
- For anything time-sensitive or outside your training data (weather, news, prices, \
live docs, recent events), use web_search to find sources, then fetch_url to read the \
most relevant one before answering. Search snippets are often stale or cached — for live \
values (prices, scores, today's figures) do NOT answer from the snippets alone; fetch_url \
the page to confirm. Do NOT claim you can't access the internet or lack a weather tool — \
look it up.

Self-improvement:
- After solving a non-trivial, reusable task, consider saving the procedure as a \
skill via skill_manage(action="create").
- When the user states a durable preference or fact, save it with memory_write."""


def build_system_prompt(skills_index: str) -> str:
    parts = [
        _BASE,
        "",
        "## Available skills",
        skills_index,
        "Call skill_view(name) to read a skill's full instructions before following it.",
    ]
    return "\n".join(parts)


def wrap_memory_context(memory_block: str) -> str:
    """Wrap prefetched memory as a reference-only block to ride the *current user turn*
    (never the system prompt), so dynamic memory can't perturb the cacheable prefix."""
    return f"<memory-context reference-only>\n{memory_block}\n</memory-context>"


def wrap_datetime_context(now: datetime) -> str:
    """Current date/time as a reference block that rides the *current user turn*, never the
    system prompt (which must stay a byte-stable cacheable prefix, S3). Local models have no
    clock and otherwise hallucinate the date from their training cutoff, so "today's …"
    queries can't be reasoned about — or searched with a correct time frame — without it."""
    return (
        f"<current-datetime reference-only>\n{now:%Y-%m-%d %H:%M %Z} ({now:%A})\n"
        "</current-datetime>"
    )
