from __future__ import annotations

from collections.abc import Callable

from .base import Tool, ToolContext


class ToolRegistry:
    """Self-registering async tool registry.

    Borrows hermes-agent's register + toolset-grouping idea but drops its config
    exposure layer: a tool is registered by importing its module (decorator side
    effect), and `schemas()` is what the agent feeds the model. No god-file, no
    20-section config.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict,
        needs_approval: bool = False,
        toolset: str = "core",
        check_fn: Callable[[ToolContext], bool] | None = None,
    ):
        def decorator(fn):
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters,
                    handler=fn,
                    needs_approval=needs_approval,
                    toolset=toolset,
                    check_fn=check_fn,
                )
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self, toolsets: set[str] | None = None) -> list[dict]:
        tools = self.all()
        if toolsets is not None:
            tools = [t for t in tools if t.toolset in toolsets]
        return [t.to_openai() for t in tools]


# Module-level singleton the builtin tool modules register against.
registry = ToolRegistry()
