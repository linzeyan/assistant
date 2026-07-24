"""Memory tools: store and recall durable facts/preferences.

Writing memory is low-risk and core to "the assistant learns about you", so it is
not approval-gated. Search is read-only.
"""

from __future__ import annotations

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="memory_write",
    description=(
        "Save a durable fact, preference, or decision to long-term memory so it can "
        "be recalled in future conversations."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["content"],
    },
)
async def memory_write(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.memory is None:
        return ToolResult(False, "memory is unavailable")
    entry = await ctx.memory.write(args["content"], args.get("tags") or [])
    return ToolResult(True, f"remembered (id={entry['id']})")


@registry.tool(
    name="memory_search",
    description="Search long-term memory for facts relevant to a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    },
)
async def memory_search(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.memory is None:
        return ToolResult(False, "memory is unavailable")
    hits = await ctx.memory.search(args["query"], limit=int(args.get("limit", 5)))
    if not hits:
        return ToolResult(True, "(no relevant memories)")
    # The id lets the model retire a stale entry via memory_forget.
    return ToolResult(True, "\n".join(f"- [{h['id']}] {h['content']}" for h in hits))


@registry.tool(
    name="memory_forget",
    description=(
        "Delete an outdated or superseded memory entry by id. Use when the user "
        "states something that replaces or contradicts a stored memory — find the "
        "old entry's id with memory_search first."
    ),
    parameters={
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "required": ["id"],
    },
    needs_approval=True,
)
async def memory_forget(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.memory is None:
        return ToolResult(False, "memory is unavailable")
    if await ctx.memory.delete(args["id"]):
        return ToolResult(True, f"forgot memory {args['id']}")
    return ToolResult(False, f"no memory with id {args['id']}")
