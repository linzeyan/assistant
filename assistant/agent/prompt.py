"""System-prompt assembly: base instructions + skills index + memory block.

Only the skills INDEX (name + description) is injected, never every skill body —
that keeps the prompt small and lets the model pull a skill's full text on demand
via skill_view. Relevant memories are prefetched per turn and appended.
"""

from __future__ import annotations

_BASE = """You are a local-first AI coding assistant running on the user's Mac.

You have tools to read/write/edit files, run shell commands, search code, search the \
web and fetch pages, manage skills, and store/recall long-term memory. Prefer using \
tools over guessing.

Current / external information:
- For anything time-sensitive or outside your training data (weather, news, prices, \
live docs, recent events), use web_search to find sources, then fetch_url to read one, \
before answering. Do NOT claim you can't access the internet or lack a weather tool — \
look it up with web_search.

Self-improvement:
- After solving a non-trivial, reusable task, consider saving the procedure as a \
skill via skill_manage(action="create").
- When the user states a durable preference or fact, save it with memory_write."""


def build_system_prompt(skills_index: str, memory_block: str) -> str:
    parts = [
        _BASE,
        "",
        "## Available skills",
        skills_index,
        "Call skill_view(name) to read a skill's full instructions before following it.",
    ]
    if memory_block:
        parts += ["", "## Relevant memory", memory_block]
    return "\n".join(parts)
