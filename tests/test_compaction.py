"""Conversation compaction (spring1 S6). The tests pin the WHY: never split a tool exchange,
never drop history on an empty summary, prefer the model's real window over the fallback, and
keep the originals recoverable.
"""

from assistant.agent.compaction import CompactionManager
from assistant.agent.session import Session
from assistant.agent.tokens import estimate_messages_tokens
from assistant.models.mlx_service import _extract_ctx, _read_context_window


class FakeLLM:
    def __init__(self, *, window=None, summary="SUMMARY TEXT"):
        self._window = window
        self._summary = summary
        self.summarize_calls: list = []

    async def context_window(self, model):
        return self._window

    def stream_chat(self, messages, model, tools=None, **params):
        self.summarize_calls.append(messages)
        summary = self._summary

        async def gen():
            if summary:
                yield {"type": "text", "content": summary}

        return gen()


def _mgr(llm, *, window_fallback=200, reserve=10, keep_recent=40):
    return CompactionManager(
        llm,
        context_window_fallback=window_fallback,
        reserve_tokens=reserve,
        keep_recent_tokens=keep_recent,
    )


def _over_budget_session():
    s = Session(id="c", model="m")
    s.messages = [{"role": "system", "content": "SYS"}]
    for k in range(8):
        s.messages.append({"role": "user", "content": f"U{k} " + "x" * 56})
        s.messages.append({"role": "assistant", "content": f"A{k} " + "y" * 56})
    return s


async def test_no_compaction_under_budget():
    s = Session(id="s", model="m")
    s.messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    before = list(s.messages)
    assert await _mgr(FakeLLM(window=100_000)).maybe_compact(s, "m") is None
    assert s.messages == before  # untouched


async def test_compacts_over_budget_and_archives_originals():
    s = _over_budget_session()
    before = estimate_messages_tokens(s.messages)
    event = await _mgr(FakeLLM()).maybe_compact(s, "m")
    assert event is not None and event["type"] == "compaction"
    # Rebuilt as [system, summary, ...recent]; system preserved, recent starts on a turn.
    assert s.messages[0] == {"role": "system", "content": "SYS"}
    assert s.messages[1]["role"] == "user" and "SUMMARY TEXT" in s.messages[1]["content"]
    assert "conversation-summary" in s.messages[1]["content"]
    assert s.messages[2]["role"] == "user"  # recent begins on a user turn
    # Recent turns are kept verbatim (the very last original message survives).
    assert s.messages[-1] == {"role": "assistant", "content": "A7 " + "y" * 56}
    # Token footprint actually shrank, and it was reported.
    after = estimate_messages_tokens(s.messages)
    assert after < before == event["tokens_before"] and event["tokens_after"] == after
    # Originals are archived — nothing is lost.
    assert len(s.compactions) == 1
    rec = s.compactions[0]
    assert rec["dropped_count"] == rec["dropped"].__len__() > 0
    assert rec["dropped"][0] == {"role": "user", "content": "U0 " + "x" * 56}


async def test_split_turn_safety_keeps_tool_exchange_together():
    s = Session(id="t", model="m")
    s.messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u0 " + "x" * 56},
        {"role": "assistant", "content": "a0 " + "y" * 56},
        {"role": "user", "content": "u1 " + "x" * 56},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "1", "type": "function",
                 "function": {"name": "read_file", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "1", "content": "t1 " + "z" * 56},
        {"role": "assistant", "content": "a1 final " + "y" * 52},
    ]
    # force_compact exercises the split logic directly (this small session is under budget).
    event = await _mgr(FakeLLM(), keep_recent=45).force_compact(s, "m")
    assert event is not None
    recent = s.messages[2:]  # after system + summary
    assert recent[0]["role"] == "user"  # boundary aligned to a turn start
    # The assistant(tool_calls) and its tool result are never separated.
    tool_idx = next(i for i, m in enumerate(recent) if m["role"] == "tool")
    assert recent[tool_idx - 1].get("tool_calls"), "tool result orphaned from its call"
    # The whole first turn (u0 + a0) was the only thing old enough to summarize.
    assert [m["content"] for m in s.compactions[0]["dropped"]] == [
        "u0 " + "x" * 56, "a0 " + "y" * 56
    ]


async def test_empty_summary_aborts_without_data_loss():
    s = _over_budget_session()
    before = list(s.messages)
    assert await _mgr(FakeLLM(summary="")).maybe_compact(s, "m") is None
    assert s.messages == before  # nothing dropped on a failed/empty summary
    assert s.compactions == []


async def test_detected_window_takes_precedence_over_fallback():
    s = _over_budget_session()  # ~300 tokens
    # Detected window is huge → under budget → no compaction, even with a tiny fallback.
    assert await _mgr(FakeLLM(window=100_000), window_fallback=50).maybe_compact(s, "m") is None
    # Window unknown → fall back to the (small) configured window → compaction triggers.
    assert await _mgr(FakeLLM(window=None), window_fallback=200).maybe_compact(s, "m")


async def test_force_compact_ignores_threshold_but_needs_old_turns():
    big = FakeLLM(window=100_000)
    # Single short turn: nothing old enough to summarize safely → None even when forced.
    short = Session(id="x", model="m")
    short.messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}]
    assert await _mgr(big).force_compact(short, "m") is None
    # A long session compacts under force despite being under the (huge) window budget.
    s = _over_budget_session()
    assert await _mgr(big).force_compact(s, "m") is not None


async def test_no_model_is_a_noop():
    s = _over_budget_session()
    assert await _mgr(FakeLLM()).maybe_compact(s, None) is None
    assert await _mgr(FakeLLM()).force_compact(s, None) is None


def test_context_window_extraction_direct_and_nested():
    assert _extract_ctx({"max_position_embeddings": 4096}) == 4096
    assert _extract_ctx({"n_ctx": 2048}) == 2048
    assert _extract_ctx({"text_config": {"max_position_embeddings": 8192}}) == 8192
    assert _extract_ctx({"hidden_size": 1}) is None


def test_read_context_window_from_config_json(tmp_path):
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 32768}')
    assert _read_context_window(tmp_path) == 32768
    assert _read_context_window(tmp_path / "missing") is None  # no file → None
