"""Prefix stability (spring1 S2+S3): the system prompt is a byte-stable cacheable prefix,
and per-turn memory rides the user turn instead of mutating that prefix. These tests guard
the WHY — on-device every cache miss re-prefills the whole prompt — not just the mechanics.
"""

import logging

from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext


class CapturingLLM:
    """Records the messages handed to each stream_chat call; answers in one text turn."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def stream_chat(self, messages, model, tools=None, **params):
        self.calls.append([dict(m) for m in messages])

        async def gen():
            yield {"type": "text", "content": "ok"}

        return gen()


class CountingMemory:
    """Returns a DIFFERENT block each turn — proves dynamic memory never perturbs the
    cacheable system prefix."""

    def __init__(self):
        self.n = 0

    async def prefetch(self, query):
        self.n += 1
        return f"- memory turn {self.n}"


def _loop(llm, tmp_path, *, memory=None, skills=None):
    return AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path, memory=memory, skills=skills),
    )


async def _run(loop, session, text, model="m"):
    return [e async for e in loop.run(session, text, model)]


def _system(call):
    return call[0]["content"]


def _last_user(call):
    return next(m for m in reversed(call) if m.get("role") == "user")["content"]


async def test_memory_rides_user_turn_not_system(tmp_path):
    llm = CapturingLLM()
    session = Session(id="s1")
    await _run(_loop(llm, tmp_path, memory=CountingMemory()), session, "hello")
    sent = llm.calls[0]
    assert "<memory-context" not in _system(sent)  # never in the cacheable prefix
    assert "<memory-context" in _last_user(sent)  # rides the current user turn
    assert "memory turn 1" in _last_user(sent)
    # Stored history stays clean — the GUI shows just what the user typed.
    stored_user = next(m for m in session.messages if m["role"] == "user")
    assert stored_user["content"] == "hello"


async def test_no_memory_leaves_user_turn_clean(tmp_path):
    llm = CapturingLLM()
    await _run(_loop(llm, tmp_path, memory=None), Session(id="s2"), "hello")
    sent = llm.calls[0]
    assert _last_user(sent) == "hello"
    assert "<memory-context" not in _system(sent)


async def test_system_prefix_byte_stable_across_turns(tmp_path, caplog):
    llm = CapturingLLM()
    session = Session(id="s3")
    loop = _loop(llm, tmp_path, memory=CountingMemory())
    with caplog.at_level(logging.WARNING, logger="assistant"):
        await _run(loop, session, "first")
        await _run(loop, session, "second")
    # Memory differs between turns, yet the system prefix is byte-for-byte identical...
    assert _system(llm.calls[0]) == _system(llm.calls[1])
    assert "memory turn 1" in _last_user(llm.calls[0])
    assert "memory turn 2" in _last_user(llm.calls[1])
    # ...and a stable model means NO cache-miss warning.
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


async def test_model_change_warns_exactly_once(tmp_path, caplog):
    llm = CapturingLLM()
    session = Session(id="s4")
    loop = _loop(llm, tmp_path)
    with caplog.at_level(logging.WARNING, logger="assistant"):
        await _run(loop, session, "a", model="m1")  # first install ("new") — no warn
        await _run(loop, session, "b", model="m1")  # reused — no warn
        await _run(loop, session, "c", model="m2")  # model swap = cache miss — one warn
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "fingerprint changed" in warns[0].getMessage()
    # The persisted fingerprint tracks the latest model.
    assert session.system_fingerprint == AgentLoop._system_fingerprint(
        loop._build_stable_system(), "m2"
    )
