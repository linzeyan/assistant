"""Unit tests for the pure reliability-measurement logic (spring5 A1).

These lock the death-mode classification and the report numbers without a live backend — the
trace dicts here are the same shape the loop records and the harness reads.
"""

from __future__ import annotations

from assistant.eval import reliability as R


def _answered(parsed_calls):
    return {"outcome": "answered", "steps": [{"parsed_calls": parsed_calls, "tool_results": []}]}


def test_classify_success_when_a_tool_was_called():
    trace = _answered([{"name": "web_search", "arguments": {"query": "x"}}])
    assert R.classify_turn(trace, expects_tool=True) == R.SUCCESS


def test_classify_no_call_is_death_mode_1_only_when_a_tool_was_expected():
    trace = _answered([])  # answered, but never called a tool
    assert R.classify_turn(trace, expects_tool=True) == R.NO_CALL
    # A prompt that didn't need a tool answering directly is a legitimate success, not mode 1.
    assert R.classify_turn(trace, expects_tool=False) == R.SUCCESS


def test_classify_reads_terminal_outcomes_straight_through():
    assert R.classify_turn({"outcome": "parse_miss"}) == R.PARSE_MISS
    assert R.classify_turn({"outcome": "tool_error"}) == R.TOOL_ERROR
    assert R.classify_turn({"outcome": "max_iters"}) == R.MAX_ITERS
    assert R.classify_turn({"outcome": "timeout"}) == R.TIMEOUT
    assert R.classify_turn({"outcome": "error"}) == R.CRASH


def test_summarize_counts_and_rate():
    buckets = [R.SUCCESS, R.SUCCESS, R.NO_CALL, R.PARSE_MISS]
    summary = R.summarize(buckets)
    assert summary["total"] == 4
    assert summary["success"] == 2
    assert summary["success_rate"] == 0.5
    # Breakdown is display-ordered and omits zero buckets.
    assert summary["breakdown"] == [(R.SUCCESS, 2), (R.NO_CALL, 1), (R.PARSE_MISS, 1)]


def test_summarize_empty_is_zero_not_division_error():
    summary = R.summarize([])
    assert summary["total"] == 0 and summary["success_rate"] == 0.0


def test_format_report_shows_rate_and_mode_labels():
    summary = R.summarize([R.SUCCESS, R.NO_CALL, R.PARSE_MISS, R.TOOL_ERROR])
    report = R.format_report(summary, model="m", runs=1)
    assert "25%" in report  # 1/4 success
    assert "m" in report
    assert "mode 1" in report and "mode 2" in report and "mode 3" in report
