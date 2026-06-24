"""Tiny in-process hook seam (spring1 P2).

Three extension points the agent loop fires, deliberately minimal — this is a seam for trusted
in-process extensions (policy, redaction, telemetry), NOT a plugin system:

  - ``before_agent_start(session, user_text)`` — observe/prepare a turn before reasoning starts.
  - ``tool_call(name, arguments) -> ToolGate | None`` — veto or mutate a call before it runs.
  - ``tool_result(name, result) -> ToolResult | None`` — observe or replace a result.

Hooks run in registration order. For ``tool_call`` the first hook that blocks wins; argument
mutations flow forward to later hooks and to the call itself. Because hooks run AFTER the
approval decision, a mutating hook is trusted code — it can change what actually executes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from assistant.tools.base import ToolResult


@dataclass
class ToolGate:
    """A ``tool_call`` hook's decision. ``block`` vetoes the call (``reason`` is surfaced to the
    model); ``arguments``, when set, replaces the call's arguments before it runs."""

    block: bool = False
    reason: str = ""
    arguments: dict | None = None


BeforeStartHook = Callable[..., Awaitable[None]]
ToolCallHook = Callable[[str, dict], Awaitable["ToolGate | None"]]
ToolResultHook = Callable[[str, ToolResult], Awaitable["ToolResult | None"]]


class HookRegistry:
    """Holds the registered hooks and fires each point. Empty by default — every fire is a
    no-op until something registers, so the seam costs nothing when unused."""

    def __init__(self) -> None:
        self._before_start: list[BeforeStartHook] = []
        self._tool_call: list[ToolCallHook] = []
        self._tool_result: list[ToolResultHook] = []

    def on_before_start(self, fn: BeforeStartHook) -> BeforeStartHook:
        self._before_start.append(fn)
        return fn

    def on_tool_call(self, fn: ToolCallHook) -> ToolCallHook:
        self._tool_call.append(fn)
        return fn

    def on_tool_result(self, fn: ToolResultHook) -> ToolResultHook:
        self._tool_result.append(fn)
        return fn

    async def fire_before_start(self, session, user_text: str) -> None:
        for fn in self._before_start:
            await fn(session, user_text)

    async def fire_tool_call(self, name: str, arguments: dict) -> ToolGate:
        gate = ToolGate(arguments=arguments)
        for fn in self._tool_call:
            decision = await fn(name, gate.arguments)
            if decision is None:
                continue
            if decision.arguments is not None:
                gate.arguments = decision.arguments  # mutation flows forward
            if decision.block:
                return ToolGate(
                    block=True,
                    reason=decision.reason or "blocked by hook",
                    arguments=gate.arguments,
                )
        return gate

    async def fire_tool_result(self, name: str, result: ToolResult) -> ToolResult:
        for fn in self._tool_result:
            replacement = await fn(name, result)
            if replacement is not None:
                result = replacement
        return result
