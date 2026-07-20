"""Token estimation (spring1 compaction prerequisite). Heuristic, but its contracts —
empty=0, ceil division, monotonic growth with history, tool_calls counted — are what
compaction's threshold logic depends on, so they're pinned here.
"""

from assistant.agent.tokens import cut_at_tokens, estimate_messages_tokens, estimate_tokens


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


def test_estimate_tokens_counts_cjk_per_char():
    # WHY (N94): non-ASCII (CJK especially) tokenizes near 1 token/char. Weighting it at
    # chars/4 under-budgeted dense scripts ~4x — one fetched CJK page slipped a ~20k-token
    # bomb past every budget built on this estimate.
    assert estimate_tokens("中" * 100) == 100
    # Mixed text: each script weighted by its own rule (8 ASCII / 4 = 2, plus 10 CJK).
    assert estimate_tokens("abcdefgh" + "字" * 10) == 12


def test_cut_at_tokens_passthrough_within_budget():
    assert cut_at_tokens("abcd", 1) == "abcd"


def test_cut_at_tokens_ascii_matches_old_char_slice():
    # 8 tokens × 4 chars/token = 32 ASCII chars kept — same arithmetic as a plain [:chars]
    # slice, so English callers behave exactly as before the token-aware cut.
    assert cut_at_tokens("a" * 100, 8) == "a" * 32


def test_cut_at_tokens_cjk_cut_four_times_sooner():
    # The same 8-token budget admits only 8 of the denser characters.
    assert cut_at_tokens("中" * 100, 8) == "中" * 8
