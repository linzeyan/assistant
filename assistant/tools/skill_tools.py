"""Skill tools: list / view existing skills, and self-author new ones.

``skill_manage`` is approval-gated — letting the agent modify its own skill library
is exactly the kind of self-modification that should pass through a human (or the
configured policy) first.
"""

from __future__ import annotations

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="skills_list",
    description="List available skills (name + one-line description).",
    parameters={"type": "object", "properties": {}},
)
async def skills_list(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.skills is None:
        return ToolResult(False, "skills are unavailable")
    return ToolResult(True, ctx.skills.index_text())


@registry.tool(
    name="skill_view",
    description="Read a skill's full instructions by name before following it.",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
)
async def skill_view(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.skills is None:
        return ToolResult(False, "skills are unavailable")
    body = ctx.skills.read_body(args["name"])
    if body is None:
        return ToolResult(False, f"unknown skill: {args['name']}")
    return ToolResult(True, body)


@registry.tool(
    name="skill_manage",
    description=(
        "Create, update, or delete a learned skill. Use this to save a reusable "
        "procedure as a new skill after solving a non-trivial task."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "update", "delete"]},
            "name": {"type": "string"},
            "description": {"type": "string", "description": "One-line summary (create/update)."},
            "body": {"type": "string", "description": "Markdown instructions (create/update)."},
        },
        "required": ["action", "name"],
    },
    needs_approval=True,
)
async def skill_manage(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.skills is None:
        return ToolResult(False, "skills are unavailable")
    # Lazy import avoids a tools<->skills import cycle at module load.
    from assistant.skills.manage import create_skill, delete_skill

    action = args["action"]
    if action in ("create", "update"):
        if not args.get("description") or not args.get("body"):
            return ToolResult(False, "create/update require both 'description' and 'body'")
        ok, msg = create_skill(
            ctx.skills,
            args["name"],
            args["description"],
            args["body"],
            overwrite=(action == "update"),
        )
        return ToolResult(ok, msg)
    if action == "delete":
        ok, msg = delete_skill(ctx.skills, args["name"])
        return ToolResult(ok, msg)
    return ToolResult(False, f"unknown action: {action}")
