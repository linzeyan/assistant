"""Interactive (GUI/HTTP) approval: the loop emits an approval_request and waits."""

from __future__ import annotations

from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.tools import build_registry
from assistant.tools.approval import (
    InteractiveApprover,
    PolicyApprover,
    resolve_pending,
)
from assistant.tools.base import ToolContext


class FakeLLM:
    def __init__(self, turns):
        self._turns = turns
        self._i = 0

    def stream_chat(self, messages, model, tools=None, **params):
        events = self._turns[self._i]
        self._i += 1

        async def gen():
            for e in events:
                yield e

        return gen()


def _loop(llm, tmp_path):
    return AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=True),
        ToolContext(cwd=tmp_path),
    )


def _write_turn():
    return [
        {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "write_file",
                    "arguments": {"path": "o.txt", "content": "data"},
                }
            ],
        }
    ]


async def test_approve_runs_the_tool(tmp_path):
    llm = FakeLLM([_write_turn(), [{"type": "text", "content": "done"}]])
    pending: dict = {}
    approver = InteractiveApprover(pending, timeout=5)

    events = []
    async for ev in _loop(llm, tmp_path).run(
        Session(id="s"), "write", "m", approver=approver
    ):
        events.append(ev)
        if ev["type"] == "approval_request":
            assert resolve_pending(pending, ev["token"], True)

    assert any(e["type"] == "approval_request" for e in events)
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and (tmp_path / "o.txt").read_text() == "data"


async def test_deny_blocks_the_tool(tmp_path):
    llm = FakeLLM([_write_turn(), [{"type": "text", "content": "ok"}]])
    pending: dict = {}
    approver = InteractiveApprover(pending, timeout=5)

    async for ev in _loop(llm, tmp_path).run(
        Session(id="s2"), "write", "m", approver=approver
    ):
        if ev["type"] == "approval_request":
            resolve_pending(pending, ev["token"], False)
        last_result = ev if ev["type"] == "tool_result" else locals().get("last_result")

    assert last_result["ok"] is False and "requires approval" in last_result["content"]
    assert not (tmp_path / "o.txt").exists()


async def test_safe_tool_skips_approval(tmp_path):
    (tmp_path / "x.txt").write_text("BODY")
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {"id": "c1", "name": "read_file", "arguments": {"path": "x.txt"}}
                    ],
                }
            ],
            [{"type": "text", "content": "done"}],
        ]
    )
    approver = InteractiveApprover({}, timeout=5)
    events = [
        ev
        async for ev in _loop(llm, tmp_path).run(
            Session(id="s3"), "read", "m", approver=approver
        )
    ]
    assert not any(e["type"] == "approval_request" for e in events)
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and tr["content"] == "BODY"


async def test_resolve_pending_unknown_token_is_false():
    assert resolve_pending({}, "nope", True) is False
