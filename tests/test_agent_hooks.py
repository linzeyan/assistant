"""Hook seam (P2): three in-process points. Tests encode WHY — a block must stop the tool from
running, a mutation must reach the handler, a result replacement must take effect, and the
first blocking tool_call hook must short-circuit the rest."""

from assistant.agent.hooks import HookRegistry, ToolGate
from assistant.agent.loop import AgentLoop
from assistant.tools.base import Tool, ToolResult


def _echo_tool(name: str = "bash") -> Tool:
    async def handler(args, ctx):
        return ToolResult(True, f"ran {args}")

    return Tool(name=name, description="", parameters={}, handler=handler, needs_approval=True)


def _loop(hooks: HookRegistry) -> AgentLoop:
    return AgentLoop(None, None, None, None, hooks=hooks)


async def test_empty_registry_is_noop():
    reg = HookRegistry()
    await reg.fire_before_start(object(), "hi")  # no hooks registered -> no error
    gate = await reg.fire_tool_call("bash", {"command": "ls"})
    assert gate.block is False and gate.arguments == {"command": "ls"}
    result = ToolResult(True, "x")
    assert await reg.fire_tool_result("bash", result) is result  # unchanged


async def test_before_start_fires_in_registration_order():
    reg = HookRegistry()
    seen: list[tuple[str, str]] = []

    @reg.on_before_start
    async def first(session, text):
        seen.append(("first", text))

    @reg.on_before_start
    async def second(session, text):
        seen.append(("second", text))

    await reg.fire_before_start(object(), "hello")
    assert seen == [("first", "hello"), ("second", "hello")]


async def test_tool_call_block_prevents_handler_running():
    reg = HookRegistry()

    @reg.on_tool_call
    async def deny(name, args):
        return ToolGate(block=True, reason="nope")

    ran: list[dict] = []

    async def handler(args, ctx):
        ran.append(args)
        return ToolResult(True, "ran")

    tool = Tool(name="bash", description="", parameters={}, handler=handler, needs_approval=True)
    result = await _loop(reg)._run_tool(tool, {"id": "1", "arguments": {"command": "ls"}}, None)
    assert result.ok is False and "blocked by hook: nope" in result.content
    assert ran == []  # handler never ran


async def test_tool_call_mutation_reaches_handler():
    reg = HookRegistry()

    @reg.on_tool_call
    async def redact(name, args):
        return ToolGate(arguments={"command": "safe"})

    result = await _loop(reg)._run_tool(
        _echo_tool(), {"id": "1", "arguments": {"command": "danger"}}, None
    )
    assert "safe" in result.content and "danger" not in result.content


async def test_tool_result_can_be_replaced():
    reg = HookRegistry()

    @reg.on_tool_result
    async def cap(name, result):
        return ToolResult(result.ok, "REPLACED")

    result = await _loop(reg)._run_tool(
        _echo_tool(), {"id": "1", "arguments": {"command": "ls"}}, None
    )
    assert result.content == "REPLACED"


async def test_first_blocking_tool_call_hook_short_circuits():
    reg = HookRegistry()
    calls: list[str] = []

    @reg.on_tool_call
    async def first(name, args):
        calls.append("first")
        return ToolGate(block=True, reason="stop")

    @reg.on_tool_call
    async def second(name, args):
        calls.append("second")
        return None

    gate = await reg.fire_tool_call("bash", {"command": "ls"})
    assert gate.block and gate.reason == "stop"
    assert calls == ["first"]  # second never runs
