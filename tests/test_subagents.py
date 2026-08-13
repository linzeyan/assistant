"""Product-native subagents (N105): parallel fan-out via spawn_subagents.

The lane/batching itself is N104's concern (test_mlx_batch); these tests cover the
runner's orchestration — fan-out, isolation, failure containment, recursion guard —
and the end-to-end wiring through the agent loop and tool registry.
"""

from __future__ import annotations

from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.agent.subagents import SubagentRunner
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext


class RoutedLLM:
    """Scripted per task keyword: the first key found in the latest user message selects
    the next event-list for that key. Records every call's params (lane-flag assertions)."""

    def __init__(self, routes: dict[str, list[list[dict]]]):
        self._routes = {k: list(v) for k, v in routes.items()}
        self.calls: list[tuple[str, dict]] = []

    def stream_chat(self, messages, model, tools=None, **params):
        last_user = next(m for m in reversed(messages) if m.get("role") == "user")["content"]
        key = next(k for k in self._routes if k in last_user)
        if key == "BOOM":
            raise RuntimeError("child model exploded")
        events = self._routes[key].pop(0)
        self.calls.append((key, params))

        async def gen():
            for e in events:
                yield e

        return gen()


def _text(content: str) -> list[dict]:
    return [{"type": "text", "content": content}]


def _ctx(tmp_path, runner=None, model="m") -> ToolContext:
    return ToolContext(cwd=tmp_path, model=model, subagents=runner)


async def test_runner_fans_out_and_reports_each_task(tmp_path):
    llm = RoutedLLM({"TASK-A": [_text("answer A")], "TASK-B": [_text("answer B")]})
    runner = SubagentRunner(llm, build_registry())
    ok, content = await runner.run(["TASK-A do x", "TASK-B do y"], _ctx(tmp_path, runner))
    assert ok
    assert "### Subagent 1 — TASK-A do x" in content and "answer A" in content
    assert "### Subagent 2 — TASK-B do y" in content and "answer B" in content
    # Every child request must ride the batch lane — that IS the parallelism (N104).
    assert llm.calls and all(p.get("concurrent") is True for _, p in llm.calls)


async def test_runner_validates_tasks_and_model(tmp_path):
    runner = SubagentRunner(RoutedLLM({}), build_registry())
    ok, msg = await runner.run([], _ctx(tmp_path, runner))
    assert not ok and "non-empty list" in msg
    ok, msg = await runner.run(["a"] * 5, _ctx(tmp_path, runner))
    assert not ok and "too many tasks" in msg
    ok, msg = await runner.run(["a"], _ctx(tmp_path, runner, model=None))
    assert not ok and "no model" in msg


async def test_child_failure_does_not_sink_siblings(tmp_path):
    llm = RoutedLLM({"TASK-A": [_text("answer A")], "BOOM": []})
    runner = SubagentRunner(llm, build_registry())
    ok, content = await runner.run(["TASK-A go", "BOOM go"], _ctx(tmp_path, runner))
    assert ok  # one success is a partial result, not a failure
    assert "answer A" in content
    assert "[failed: child model exploded]" in content


async def test_progress_ticks_per_completed_subagent(tmp_path):
    llm = RoutedLLM({"TASK-A": [_text("a")], "TASK-B": [_text("b")]})
    runner = SubagentRunner(llm, build_registry())
    ticks: list[tuple[float, str]] = []
    ctx = _ctx(tmp_path, runner)
    ctx.on_progress = lambda fraction, label="": ticks.append((fraction, label))
    ok, _ = await runner.run(["TASK-A", "TASK-B"], ctx)
    assert ok
    assert sorted(f for f, _ in ticks) == [0.5, 1.0]
    assert all("subagents done" in label for _, label in ticks)


async def test_recursion_guard_hides_tool_from_children(tmp_path):
    registry = build_registry()
    llm = RoutedLLM({})
    runner = SubagentRunner(llm, registry)
    loop = AgentLoop(llm, registry, PolicyApprover(approval_required=False),
                     _ctx(tmp_path, runner))
    with_runner = [t["function"]["name"] for t in loop._visible_tool_schemas(_ctx(tmp_path, runner))]
    without = [t["function"]["name"] for t in loop._visible_tool_schemas(_ctx(tmp_path, None))]
    assert "spawn_subagents" in with_runner
    assert "spawn_subagents" not in without


async def test_end_to_end_through_agent_loop(tmp_path):
    # Main turn calls spawn_subagents; two children answer in parallel; the main model
    # then wraps up. Verifies the whole chain: ctx.model plumbing (run() sets it), tool
    # dispatch, runner fan-out, and the combined tool_result the model sees.
    llm = RoutedLLM({
        "fan out": [
            [{"type": "tool_calls", "tool_calls": [{
                "id": "c1", "name": "spawn_subagents",
                "arguments": {"tasks": ["TASK-A annotate one", "TASK-B annotate two"]},
            }]}],
            _text("both files handled"),
        ],
        "TASK-A": [_text("A done")],
        "TASK-B": [_text("B done")],
    })
    registry = build_registry()
    runner = SubagentRunner(llm, registry)
    ctx = ToolContext(cwd=tmp_path, subagents=runner)  # model arrives via run(), not here
    loop = AgentLoop(llm, registry, PolicyApprover(approval_required=False), ctx)
    events = [e async for e in loop.run(Session(id="s1"), "please fan out the work", "m")]

    results = [e for e in events if e["type"] == "tool_result"]
    assert len(results) == 1 and results[0]["ok"]
    assert "A done" in results[0]["content"] and "B done" in results[0]["content"]
    assert events[-1]["type"] == "done"
    # Children rode the lane; the main turn did not (it is the interactive conversation).
    child_params = [p for k, p in llm.calls if k.startswith("TASK-")]
    main_params = [p for k, p in llm.calls if k == "fan out"]
    assert child_params and all(p.get("concurrent") is True for p in child_params)
    assert all("concurrent" not in p for p in main_params)
