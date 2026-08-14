"""Plan/TODO tool: a per-turn checklist the agent maintains while working a multi-step task.

Why a tool and not just a prose preamble: on small local models a visible, structured plan keeps
the agent on track across a long tool chain (read → fix → test) far better than a one-off ETHOS
line, which is mostly inert (spring4 SA.3). ``update_plan`` is a FULL REPLACE — the agent re-sends
the entire list each call — so there is no patch/merge semantics to get wrong.

Lifecycle (spring4 SA.3 必補#1): the plan is TURN-SCOPED. This tool only validates + acks; the
agent loop keeps the authoritative list as a turn-local and re-emits it as a ``plan`` event for the
UI. The full checklist is deliberately NOT written back into ``session.messages`` — re-feeding a
stale checklist on every iteration bloats history and burns context on small models (and works
against compaction, S6). The short ack returned here IS what the conversation history sees.
"""

from __future__ import annotations

import ast
import json

from .base import ToolContext, ToolResult
from .registry import registry

VALID_STATUS = ("pending", "in_progress", "completed")


def normalize_steps(raw) -> list[dict]:
    """Validate + canonicalize ``update_plan``'s ``steps`` (a list of ``{title, status}``).

    Fail loud (ValueError) on anything unusable so the tool returns a clear error instead of
    emitting a junk checklist. An unknown/missing status defaults to ``pending``; titles are
    stripped and empty-title steps dropped.
    """
    if isinstance(raw, str):
        # Small models routinely serialize a nested argument as a string instead of as JSON,
        # and often as a Python repr (single quotes) rather than JSON — which is why
        # literal_eval backs json up. Rejecting that spelling cost a real coding turn: the
        # model retried the same shape three times, then abandoned the task. The plan is an
        # aid, so meeting it where it is beats being right about the schema.
        for parse in (json.loads, ast.literal_eval):
            try:
                raw = parse(raw)
                break
            except (ValueError, SyntaxError):
                continue
    if not isinstance(raw, list) or not raw:
        raise ValueError("steps must be a non-empty array of {title, status}")
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each step must be an object with a title and status")
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUS:
            status = "pending"
        out.append({"title": title, "status": status})
    if not out:
        raise ValueError("steps must contain at least one titled step")
    return out


def plan_summary(steps: list[dict]) -> str:
    done = sum(1 for s in steps if s["status"] == "completed")
    return f"{done}/{len(steps)} done"


@registry.tool(
    name="update_plan",
    description=(
        "Maintain a short checklist for a multi-step task. Call it once at the start with the "
        "planned steps, then call it again to update statuses as you work (mark a step "
        "'in_progress' before doing it, 'completed' when done). Always send the COMPLETE list — "
        "it replaces the previous plan. Use it for tasks needing several tool calls; skip it for "
        "trivial one-step requests."
    ),
    parameters={
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "status": {"type": "string", "enum": list(VALID_STATUS)},
                    },
                    "required": ["title", "status"],
                },
            },
        },
        "required": ["steps"],
    },
)
async def update_plan(args: dict, ctx: ToolContext) -> ToolResult:
    try:
        steps = normalize_steps(args.get("steps"))
    except ValueError as exc:
        return ToolResult(False, f"invalid plan: {exc}")
    # Short ack only. The loop owns the authoritative plan and emits the `plan` event; the full
    # checklist intentionally stays OUT of session.messages (see module docstring).
    return ToolResult(True, f"plan updated ({plan_summary(steps)})")
