"""Per-turn trace (spring2 P0). The point is measure-before-fix: record each turn so the
reliability failure modes ("model didn't call / parser missed / tool errored") become a
scannable list. These pin the classification and that the loop records as a pure side
channel (the SSE event stream is unchanged)."""

from __future__ import annotations

import pytest

from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.agent.trace import (
    TraceStep,
    TraceStore,
    TurnTrace,
    looks_like_tool_attempt,
)
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext


class FakeLLM:
    """Scripted LLM: one preset event list per call (one call == one loop iteration)."""

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


def _loop(llm, tmp_path, store) -> AgentLoop:
    return AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
        trace_store=store,
    )


# --- TraceStore persistence ---


def test_store_record_get_list_roundtrip(tmp_path):
    store = TraceStore(tmp_path / "traces")
    t = TurnTrace.new("sess", "m", "查天氣").finalize("answered")
    store.record(t)
    assert store.get(t.turn_id)["user_text"] == "查天氣"  # CJK survives the JSON round-trip
    rows = store.list_for_session("sess")
    assert len(rows) == 1 and rows[0]["turn_id"] == t.turn_id
    # A fresh store (simulating a restart) reads the same trace from disk.
    assert TraceStore(tmp_path / "traces").get(t.turn_id)["user_text"] == "查天氣"


def test_store_memory_only_without_dir():
    store = TraceStore(None)
    t = TurnTrace.new("s", "m", "hi")
    store.record(t)
    assert store.get(t.turn_id)["session_id"] == "s"
    assert store.list_for_session("s")[0]["turn_id"] == t.turn_id
    assert store.get("nope") is None


def test_list_filters_by_session_and_orders_newest_first(tmp_path):
    store = TraceStore(tmp_path)
    store.record(TurnTrace("a", "s1", "m", "first", created_at=1.0))
    store.record(TurnTrace("b", "s1", "m", "second", created_at=2.0))
    store.record(TurnTrace("c", "s2", "m", "other", created_at=3.0))
    rows = store.list_for_session("s1")
    assert [r["turn_id"] for r in rows] == ["b", "a"]  # newest first; s2 excluded


def test_record_prunes_to_max_turns(tmp_path):
    store = TraceStore(tmp_path, max_turns=2)
    for i in range(4):
        store.record(TurnTrace(f"t{i}", "s", "m", "x", created_at=float(i)))
    assert len(list((tmp_path).glob("*.json"))) == 2  # oldest two pruned off disk


# --- outcome classification (the scannable signal) ---


def test_finalize_answered():
    t = TurnTrace.new("s", "m", "hi")
    t.steps.append(TraceStep(model_text="hello"))
    assert t.finalize("answered").outcome == "answered"


def test_finalize_parse_miss_on_leaked_markup():
    # Model emitted a tool call but it leaked back as text (parser missed it) — the N3 class.
    t = TurnTrace.new("s", "m", "查天氣")
    t.steps.append(
        TraceStep(model_text='<tool_call>{"name": "web_search"}</tool_call>', parsed_calls=[])
    )
    assert t.finalize("answered").outcome == "parse_miss"


def test_finalize_tool_error():
    t = TurnTrace.new("s", "m", "x")
    t.steps.append(
        TraceStep(
            parsed_calls=[{"name": "web_search", "arguments": {}}],
            tool_results=[{"name": "web_search", "ok": False, "content": "timeout"}],
        )
    )
    t.steps.append(TraceStep(model_text="sorry, search failed"))
    assert t.finalize("answered").outcome == "tool_error"


def test_finalize_max_iters_wins_over_step_signals():
    t = TurnTrace.new("s", "m", "x")
    t.steps.append(
        TraceStep(
            parsed_calls=[{"name": "x", "arguments": {}}],
            tool_results=[{"name": "x", "ok": True, "content": "ok"}],
        )
    )
    assert t.finalize("max_iters").outcome == "max_iters"


def test_parse_miss_precedes_tool_error():
    # A missed call means the tool never ran, so it's the more upstream cause to flag.
    t = TurnTrace.new("s", "m", "x")
    t.steps.append(
        TraceStep(
            parsed_calls=[{"name": "a", "arguments": {}}],
            tool_results=[{"name": "a", "ok": False, "content": "err"}],
        )
    )
    t.steps.append(TraceStep(model_text="<function=web_search></function>", parsed_calls=[]))
    assert t.finalize("answered").outcome == "parse_miss"


def test_looks_like_tool_attempt():
    assert looks_like_tool_attempt("blah <tool_call>{}")
    assert looks_like_tool_attempt("<function=foo></function>")
    assert not looks_like_tool_attempt("just a normal answer about 天氣")
    # A complete answer ending in a stray, empty <tool_call> opener is NOT an attempt — flagging it
    # mislabels an answered turn as parse_miss (real A1 capture: Qwen3-Coder dangling marker).
    assert not looks_like_tool_attempt("The record holder is Usain Bolt, 9.58s.\n<tool_call>")
    assert not looks_like_tool_attempt("done.\n<tool_call>   \n")  # only whitespace after


# --- loop integration: records as a pure side channel ---


async def test_loop_records_turn_trace(tmp_path):
    (tmp_path / "x.txt").write_text("FILE BODY")
    store = TraceStore(None)
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
    events = await _collect(_loop(llm, tmp_path, store).run(Session(id="s1"), "read x.txt", "m"))

    # SSE stream is unchanged by tracing (pure side channel).
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "assistant_delta", "done"]

    trace = store.get(store.list_for_session("s1")[0]["turn_id"])
    assert trace["outcome"] == "answered" and trace["user_text"] == "read x.txt"
    assert len(trace["steps"]) == 2
    assert trace["steps"][0]["parsed_calls"][0]["name"] == "read_file"
    assert trace["steps"][0]["tool_results"][0] == {
        "name": "read_file",
        "ok": True,
        "content": "FILE BODY",
    }
    assert trace["final_text"] == "the file says FILE BODY"


async def test_loop_flags_parse_miss_end_to_end(tmp_path):
    store = TraceStore(None)
    # Simulates mlx_service flushing unparsed tool markup back as text (a real parse miss).
    llm = FakeLLM([[{"type": "text", "content": '<tool_call>{"name": "web_search"}</tool_call>'}]])
    await _collect(_loop(llm, tmp_path, store).run(Session(id="pm"), "查台北天氣", "m"))
    trace = store.get(store.list_for_session("pm")[0]["turn_id"])
    assert trace["outcome"] == "parse_miss"


async def test_loop_records_error_on_mid_loop_exception(tmp_path):
    # The single most common real failure: a turn dies mid-loop (e.g. chat-template render
    # error feeding tool_calls back). P0 must capture it AND re-raise so the API still errors.
    store = TraceStore(None)

    class BoomLLM:
        def stream_chat(self, messages, model, tools=None, **params):
            async def gen():
                raise RuntimeError("Can only get item pairs from a mapping.")
                yield  # unreachable, but makes this an async generator

            return gen()

    loop = AgentLoop(
        BoomLLM(),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
        trace_store=store,
    )
    with pytest.raises(RuntimeError):  # outward behaviour unchanged — the error propagates
        await _collect(loop.run(Session(id="boom"), "查台北天氣", "m"))

    trace = store.get(store.list_for_session("boom")[0]["turn_id"])
    assert trace["outcome"] == "error"
    assert "item pairs" in trace["error"]


async def test_loop_without_trace_store_is_noop(tmp_path):
    llm = FakeLLM([[{"type": "text", "content": "hi"}]])
    loop = AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
    )  # no trace_store
    events = await _collect(loop.run(Session(id="s"), "hi", "m"))
    assert events[-1]["type"] == "done"  # runs fine, just records nothing


def test_trace_routes_are_wired(tmp_path):
    # Proves the router is registered and app.state.trace_store is present — the endpoints
    # the user actually hits to scan turns. trace_dir → tmp so it doesn't touch real dirs.
    from fastapi.testclient import TestClient

    from assistant.config import Settings
    from assistant.main import create_app

    app = create_app(
        Settings(models_dir=tmp_path / "models", trace_dir=tmp_path / "traces")
    )
    with TestClient(app) as client:
        assert client.get("/turns/nope").status_code == 404
        r = client.get("/sessions/whatever/turns")
        assert r.status_code == 200 and r.json() == {"turns": []}
