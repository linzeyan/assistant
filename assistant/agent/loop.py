from __future__ import annotations

import json
from collections.abc import AsyncIterator

from assistant.tools.approval import ApprovalPolicy
from assistant.tools.base import ToolContext, ToolResult
from assistant.tools.registry import ToolRegistry

from .llm_client import AsyncLLM
from .prompt import build_system_prompt
from .session import Session


class AgentLoop:
    """The think -> tool -> observe cycle.

    Each iteration streams one assistant turn. Text deltas are forwarded to the
    caller; if the model emits tool calls, each is approval-checked, dispatched, its
    result appended to the conversation, and the loop runs again. When a turn
    produces no tool calls, that turn is the final answer.

    `run()` yields typed event dicts (assistant_delta / tool_call / tool_result /
    done / error) which the API layer forwards verbatim as SSE.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        registry: ToolRegistry,
        approver: ApprovalPolicy,
        ctx: ToolContext,
        max_iters: int = 8,
    ):
        self._llm = llm
        self._registry = registry
        self._approver = approver
        self._ctx = ctx
        self._max_iters = max_iters

    async def run(
        self,
        session: Session,
        user_text: str,
        model: str,
        approver: ApprovalPolicy | None = None,
    ) -> AsyncIterator[dict]:
        # Per-run approver lets the Telegram gateway inject an interactive (inline
        # button) approver while the desktop path uses the default policy approver.
        effective_approver = approver or self._approver
        session.set_system(await self._build_system_prompt(user_text))
        session.add_user(user_text)
        tools = self._registry.schemas() or None

        for _ in range(self._max_iters):
            text_parts: list[str] = []
            tool_calls: list[dict] | None = None

            async for ev in self._llm.stream_chat(session.messages, model, tools=tools):
                if ev["type"] == "text":
                    text_parts.append(ev["content"])
                    yield {"type": "assistant_delta", "content": ev["content"]}
                elif ev["type"] == "tool_calls":
                    tool_calls = ev["tool_calls"]

            if not tool_calls:
                session.add_assistant("".join(text_parts))
                yield {"type": "done"}
                return

            session.messages.append(self._assistant_tool_msg(text_parts, tool_calls))
            for tc in tool_calls:
                yield {
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                tool = self._registry.get(tc["name"])
                if tool is None:
                    result = ToolResult(False, f"unknown tool: {tc['name']}")
                elif tool.needs_approval and getattr(
                    effective_approver, "interactive", False
                ):
                    # Interactive approver (GUI/HTTP): surface the request and wait for
                    # an out-of-band decision (POST /chat/approve) before running.
                    token = effective_approver.new_request()
                    yield {
                        "type": "approval_request",
                        "id": tc["id"],
                        "token": token,
                        "name": tool.name,
                        "arguments": tc["arguments"],
                    }
                    if await effective_approver.wait(token):
                        result = await self._run_tool(tool, tc)
                    else:
                        result = ToolResult(
                            False, f"denied: tool '{tool.name}' requires approval"
                        )
                elif not await effective_approver.approve(tool, tc["arguments"]):
                    result = ToolResult(
                        False, f"denied: tool '{tool.name}' requires approval"
                    )
                else:
                    result = await self._run_tool(tool, tc)
                session.messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
                )
                yield {
                    "type": "tool_result",
                    "id": tc["id"],
                    "name": tc["name"],
                    "ok": result.ok,
                    "content": result.content,
                }

        yield {"type": "error", "detail": f"reached max tool iterations ({self._max_iters})"}

    async def _build_system_prompt(self, user_text: str) -> str:
        # Rebuilt each turn so memory prefetch reflects the latest message. Returned
        # (not stored on self) so concurrent runs — e.g. several Telegram chats — do
        # not race on shared loop state. Skills/memory live on the tool context.
        skills = self._ctx.skills
        memory = self._ctx.memory
        skills_index = skills.index_text() if skills else "(no skills available)"
        memory_block = await memory.prefetch(user_text) if memory else ""
        return build_system_prompt(skills_index, memory_block)

    @staticmethod
    def _assistant_tool_msg(text_parts: list[str], tool_calls: list[dict]) -> dict:
        return {
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }

    async def _run_tool(self, tool, tc: dict) -> ToolResult:
        try:
            return await tool.handler(tc["arguments"], self._ctx)
        except Exception as exc:  # never let a tool crash take down the turn
            return ToolResult(False, f"error running {tool.name}: {exc}")
