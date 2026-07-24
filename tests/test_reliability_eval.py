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


def _tool_error_trace(results):
    """A turn the loop marked tool_error — which it only does when the turn DID reach an answer."""
    return {
        "outcome": "tool_error",
        "steps": [
            {"parsed_calls": [{"name": n}], "tool_results": [{"name": n, "ok": ok}]}
            for n, ok in results
        ],
    }


def test_recovered_when_the_model_routed_around_a_failed_call():
    # The real shape that made a 15/16 sweep report 38%: web tools worked, the model then tried a
    # tool that failed, and still answered. That turn delivered — counting it as death mode 3 hides
    # a working product behind a scary number.
    trace = _tool_error_trace([("web_search", True), ("find", False), ("web_search", True)])
    assert R.classify_turn(trace) == R.RECOVERED


def test_tool_error_stays_fatal_when_nothing_ever_worked():
    # Every call failed and the model answered anyway — it answered from nothing, which is the
    # failure mode 3 is meant to catch. Recovery requires that some tool actually returned data.
    trace = _tool_error_trace([("fetch_url", False), ("fetch_url", False)])
    assert R.classify_turn(trace) == R.TOOL_ERROR


def _row(prompt: str, delivered: bool) -> dict:
    """A capture-shaped row (what measure --append-corpus writes) for one run of a prompt."""
    calls = [{"name": "web_search", "arguments": {}}] if delivered else []
    return {
        "user_text": prompt,
        "outcome": "answered",
        "steps": [{"parsed_calls": calls, "tool_results": []}],
    }


def test_compare_exposes_the_hidden_trade():
    # WHY (AHE transition matrix): both sweeps deliver 2/4 — the headline says "unchanged" —
    # but prompt A was fixed and prompt B broken. The per-prompt matrix must expose exactly
    # that, with the regression surfaced first.
    baseline = [_row("A", False)] * 2 + [_row("B", True)] * 2
    candidate = [_row("A", True)] * 2 + [_row("B", False)] * 2
    rows = R.compare_by_prompt(baseline, candidate)
    assert [(r["prompt"], r["transition"]) for r in rows] == [
        ("B", R.PASS_TO_FAIL),  # regression first — it's the loud one
        ("A", R.FAIL_TO_PASS),
    ]
    report = R.format_comparison(rows)
    assert "pass_to_fail" in report and "0/2 → 2/2" in report
    assert "1 prompt(s) regressed" in report


def test_compare_same_side_movement_and_missing_prompts():
    baseline = [_row("A", True), _row("A", False), _row("C", True)]
    candidate = [_row("A", True), _row("A", True)]  # A improves within pass; C vanished
    by = {r["prompt"]: r["transition"] for r in R.compare_by_prompt(baseline, candidate)}
    assert by == {"A": R.IMPROVED, "C": R.BASELINE_ONLY}
    # No regressions -> the report must say so instead of warning.
    assert "no per-prompt regressions" in R.format_comparison(R.compare_by_prompt(baseline, candidate))


def test_recovered_counts_as_delivered_but_stays_visible():
    # Both must be true at once: the headline rate reflects what the user got, AND a run carried
    # by recoveries is distinguishable from a clean one.
    summary = R.summarize([R.SUCCESS, R.RECOVERED, R.RECOVERED, R.MAX_ITERS])
    assert summary["delivered"] == 3
    assert summary["success_rate"] == 0.75
    assert summary["success"] == 1 and summary["recovered"] == 2
    report = R.format_report(summary, model="m", runs=1)
    assert "75%" in report
    assert "1 clean, 2 after a failed call" in report
    # Recovered turns are not failures — they must not appear in the death-mode list.
    assert "recovered (answered" not in report.split("failures by death mode:")[1]


def test_classify_reads_terminal_outcomes_straight_through():
    assert R.classify_turn({"outcome": "parse_miss"}) == R.PARSE_MISS
    assert R.classify_turn({"outcome": "tool_error"}) == R.TOOL_ERROR
    assert R.classify_turn({"outcome": "max_iters"}) == R.MAX_ITERS
    assert R.classify_turn({"outcome": "timeout"}) == R.TIMEOUT
    assert R.classify_turn({"outcome": "error"}) == R.CRASH


class _FakeModelsClient:
    """Just enough httpx.AsyncClient to answer the harness's /models probe."""

    def __init__(self, models):
        self._models = models

    async def get(self, _path):
        import types as _t

        return _t.SimpleNamespace(json=lambda: {"models": self._models})


async def test_autodetect_never_falls_back_to_the_fusion_panel():
    # After a crash leaves nothing loaded, the fallback used to take models[0] — which is the
    # virtual `fusion` entry. Measuring it loads the entire panel at once and OOM-kills the
    # backend, so a sweep that already died once would immediately kill it again.
    from assistant.eval.measure import _autodetect_model

    models = [
        {"id": "fusion", "type": "llm", "source": "virtual", "loaded": False},
        {"id": "some/image-model", "type": "image", "source": "local", "loaded": False},
        {"id": "some/chat-model", "type": "llm", "source": "local", "loaded": False},
    ]
    assert await _autodetect_model(_FakeModelsClient(models)) == "some/chat-model"


async def test_autodetect_returns_none_when_only_unusable_models_exist():
    # Better to tell the user to pass --model than to silently measure an image model.
    from assistant.eval.measure import _autodetect_model

    models = [
        {"id": "fusion", "type": "llm", "source": "virtual", "loaded": True},
        {"id": "some/image-model", "type": "image", "source": "local", "loaded": True},
    ]
    assert await _autodetect_model(_FakeModelsClient(models)) is None


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
