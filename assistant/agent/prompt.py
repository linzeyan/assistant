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

Ground answers in what's actually there — check, don't guess:
- When the user names a local file or directory (a path like /Users/…, ~/…, ./x, or "look \
at / check / 看下 <file>"), READ it with a tool (read_file, or bash cat/ls) BEFORE you answer, \
edit it, or write anything about it. Never describe or rewrite a file you have not opened.
- Do not fabricate shell commands, package names, install/run steps, flags, or code APIs. If \
you are not certain something is real, verify it — read the file, fetch the project's \
README/docs with fetch_url, or run `--help` — instead of guessing. A confident wrong command \
wastes the user's time and breaks trust; checking first is what they want.
- Answer the specific question in its real context (their project, their file). Do not pad the \
reply with generic alternatives, unrelated tools, or boilerplate the user did not ask for.

Current / external information:
- The current date is provided with each user message — use it; do not guess the date \
from your training data.
- For anything time-sensitive or outside your training data (weather, news, prices, \
live docs, recent events), use web_search to find sources, then fetch_url to read the \
most relevant one before answering. Search snippets are often stale or cached — for live \
values (prices, scores, today's figures) do NOT answer from the snippets alone; fetch_url \
the page to confirm. Do NOT claim you can't access the internet or lack a weather tool — \
look it up.

Media generation:
- When the user asks you to CREATE an image or a video from a description, you MUST actually \
call the matching tool (generate_image / generate_video) — issuing that tool call is the ONLY \
thing that produces the file. Announcing that you will use it, or writing the scene out in \
prose, produces NOTHING for the user. So do not describe the result, give a tutorial, or \
output a script for other software: emit the generate_image / generate_video tool call as your \
action, passing the user's request as the prompt. "generate / draw / make a video of X" means \
call the tool now.

Using skills:
- The "Available skills" list below names reusable workflows for specific situations. When a \
request clearly matches one — for example a debugging ask ("why is this broken", "it's \
erroring", "it crashed", "this test fails") matches the investigate skill — call \
skill_view(name) FIRST and follow that workflow step by step, instead of improvising an \
ad-hoc answer.

Working on multi-step tasks:
- For a task that needs several steps or tool calls, call update_plan first with a short \
checklist, then update it as you go (mark a step in_progress before working it, completed when \
done). Send the whole list each time. Skip the plan for trivial one-step requests.
- How to work: investigate before you change (find the cause, read the relevant code, don't \
patch symptoms); finish the whole task rather than stopping half-done; and when a real choice or \
a destructive/irreversible action comes up, surface it and let the user decide instead of \
guessing.

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


def wrap_workspace_context(cwd) -> str:
    """The agent's working directory as a reference block riding the *current user turn* (never
    the cacheable system prefix, S3). Without it a local model has no idea where it is, so a
    bare "git diff" / file request gets a guessed path (observed: a model ran `cd /Users/Shared
    && ls -la` instead of running git in the workspace) rather than executing in the workspace."""
    return (
        f"<workspace reference-only>\nWorking directory: {cwd}\n"
        "Run shell and file commands here; relative paths resolve against it.\n"
        "</workspace>"
    )


def wrap_referenced_paths(paths: list[str]) -> str:
    """Existing local paths the user named this turn, surfaced as a reference block riding the
    *current user turn* (never the cacheable prefix, S3). "看下 <path>" is the strongest possible
    signal to READ before answering, yet a weak-at-tools model often answers from imagination
    instead — observed: asked to "看下 …/Makefile" it fabricated a whole Makefile without ever
    opening the real 15.5K one. Naming the real paths in-context, at the exact moment, nudges it
    to read them first. Existence-gated upstream, so it only ever lists files/dirs that are there."""
    listed = "\n".join(f"- {p}" for p in paths)
    return (
        "<referenced-paths reference-only>\n"
        "The user's message points at these local paths (they exist on disk):\n"
        f"{listed}\n"
        "Open them with read_file (or bash cat/ls) BEFORE answering about, editing, or writing "
        "instructions for them — do not guess their contents.\n"
        "</referenced-paths>"
    )


def wrap_datetime_context(now: datetime) -> str:
    """Current date/time as a reference block that rides the *current user turn*, never the
    system prompt (which must stay a byte-stable cacheable prefix, S3). Local models have no
    clock and otherwise hallucinate the date from their training cutoff, so "today's …"
    queries can't be reasoned about — or searched with a correct time frame — without it."""
    return (
        f"<current-datetime reference-only>\n{now:%Y-%m-%d %H:%M %Z} ({now:%A})\n"
        "</current-datetime>"
    )
