"""Token estimation (spring1 compaction prerequisite). Heuristic, but its contracts —
empty=0, ceil division, monotonic growth with history, tool_calls counted — are what
compaction's threshold logic depends on, so they're pinned here.
"""

from assistant.agent.tokens import estimate_messages_tokens, estimate_tokens


def test_estimate_tokens_empty_is_zero():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_ceil_division():
    assert estimate_tokens("abcd") == 1  # 4 chars / 4
    assert estimate_tokens("abcde") == 2  # 5 chars / 4 -> ceil 2


def test_estimate_messages_counts_content_plus_overhead():
    # 1 token of content + the per-message overhead (4).
    assert estimate_messages_tokens([{"role": "user", "content": "abcd"}]) == 5


def test_estimate_messages_includes_tool_calls():
    base = [{"role": "user", "content": "hi"}]
    with_tc = base + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x"}'},
                }
            ],
        }
    ]
    assert estimate_messages_tokens(with_tc) > estimate_messages_tokens(base)


def test_estimate_messages_grows_with_history():
    short = [{"role": "user", "content": "hi"}]
    longer = short + [{"role": "assistant", "content": "a noticeably longer answer here"}]
    assert estimate_messages_tokens(longer) > estimate_messages_tokens(short)
