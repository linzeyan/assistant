"""spawn_subagents — parallel fan-out of independent tasks (N105).

Registration-only module (imported by ``build_registry``); the actual orchestration
lives in ``assistant.agent.subagents.SubagentRunner``, reached through the tool
context. The ``check_fn`` gate doubles as the recursion guard: subagent children run
with ``ctx.subagents=None``, so the tool never appears in their schema.
"""

from __future__ import annotations

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="spawn_subagents",
    description=(
        "Run several INDEPENDENT tasks in parallel, each in its own subagent with a fresh "
        "context and the standard tools (read/write/edit files, bash, search, web). Each "
        "task string must be complete and self-contained — include absolute file paths and "
        "the exact goal, because the subagent sees nothing else of this conversation. "
        "Returns one result summary per task. Use for independent chunks of work (e.g. "
        "process several files at once); do NOT use for steps that depend on each other. "
        "Max 4 tasks per call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One complete, self-contained instruction per subagent (max 4).",
            }
        },
        "required": ["tasks"],
    },
    needs_approval=True,  # one human gate for the whole fan-out; children don't re-prompt
    check_fn=lambda ctx: ctx.subagents is not None,
)
async def spawn_subagents(arguments: dict, ctx: ToolContext) -> ToolResult:
    if ctx.subagents is None:  # child context, or a deployment without the runner wired
        return ToolResult(False, "subagents are not available here")
    ok, content = await ctx.subagents.run(arguments.get("tasks"), ctx)
    return ToolResult(ok, content)
