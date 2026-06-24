from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext


class FakeLLM:
    """Scripted LLM: returns a preset event list per call (one call == one turn)."""

    def __init__(self, turns: list[list[dict]]):
        self._turns = turns
        self._i = 0

    def stream_chat(self, messages, model, tools=None, **params):
        events = self._turns[self._i]
        self._i += 1

        async def gen():
            for e in events:
                yield e

        return gen()


async def _collect(agen):
    return [e async for e in agen]


def _loop(llm, tmp_path, approval_required: bool) -> AgentLoop:
    return AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=approval_required),
        ToolContext(cwd=tmp_path),
    )


async def test_tool_then_final_answer(tmp_path):
    (tmp_path / "x.txt").write_text("FILE BODY")
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
            [{"type": "text", "content": "the file says FILE BODY"}],
        ]
    )
    session = Session(id="s1")
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        session, "read x.txt", "m"
    ))

    assert [e["type"] for e in events] == [
        "tool_call",
        "tool_result",
        "assistant_delta",
        "done",
    ]
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and tr["content"] == "FILE BODY"
    # Tool output and final answer are persisted so the next turn has context.
    assert any(m.get("role") == "tool" and m["content"] == "FILE BODY" for m in session.messages)
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "the file says FILE BODY",
    }


async def test_approval_gate_blocks_mutation(tmp_path):
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "data"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "could not write"}],
        ]
    )
    # approval_required=True + non-interactive approver => denied.
    events = await _collect(_loop(llm, tmp_path, approval_required=True).run(
        Session(id="s2"), "write a file", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False and "requires approval" in tr["content"]
    assert not (tmp_path / "out.txt").exists()  # mutation actually blocked


async def test_approval_disabled_allows_mutation(tmp_path):
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "data"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "wrote it"}],
        ]
    )
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        Session(id="s3"), "write a file", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and (tmp_path / "out.txt").read_text() == "data"


async def test_unknown_tool_reported(tmp_path):
    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "nope", "arguments": {}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        Session(id="s4"), "x", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False and "unknown tool" in tr["content"]


async def test_forwards_max_output_tokens_to_llm(tmp_path):
    """The configured per-turn ceiling must actually reach the engine — the engine's own
    default was 1024, which silently truncated long answers (bug #1)."""
    captured: dict = {}

    class CapturingLLM:
        def stream_chat(self, messages, model, tools=None, **params):
            captured.update(params)

            async def gen():
                yield {"type": "text", "content": "hi"}

            return gen()

    loop = AgentLoop(
        CapturingLLM(),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
        max_output_tokens=2048,
    )
    await _collect(loop.run(Session(id="mt"), "hello", "m"))
    assert captured.get("max_tokens") == 2048
