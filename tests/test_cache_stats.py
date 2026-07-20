"""Unit tests for the pure KV-cache hit-rate logic (spring6 K2).

The sample lines below are the exact N76 format `MlxEngine.stream_text` logs — if that
format drifts, these tests fail before the K2 report silently reads as "no data".
"""

from __future__ import annotations

from assistant.eval import cache_stats as C

_LOG = """\
2026-07-19 20:18:02 INFO [req 1a2b] generation: prompt=2457 (cached=0, prefill=2457) prefill=1.98s decode=1174 tok in 14.05s (83.6 tok/s)
2026-07-19 20:18:40 INFO [req 1a2b] generation: prompt=3700 (cached=3631, prefill=69) prefill=0.21s decode=90 tok in 1.10s (81.8 tok/s)
noise line that must be ignored
2026-07-19 20:19:01 INFO [req 9f00] generation: prompt=843 (cached=0, prefill=843) prefill=0.70s decode=40 tok in 0.50s (80.0 tok/s)
"""


def test_parse_extracts_records_and_skips_noise():
    records = C.parse_generation_lines(_LOG)
    assert len(records) == 3
    assert records[0] == {
        "prompt": 2457,
        "cached": 0,
        "prefill": 2457,
        "prefill_s": 1.98,
        "decode": 1174,
        "decode_s": 14.05,
        "tps": 83.6,
    }
    assert records[1]["cached"] == 3631


def test_summarize_hit_rate_and_warm_cold_split():
    s = C.summarize(C.parse_generation_lines(_LOG))
    assert s["turns"] == 3
    assert s["warm_turns"] == 1 and s["cold_or_rebuilt_turns"] == 2
    # Overall rate counts cold prefills against the cache; the warm-only rate isolates
    # how well prefix reuse works when it engages at all (the K2 drift question).
    assert s["hit_rate"] == 3631 / (2457 + 3700 + 843)
    assert s["warm_hit_rate"] == 3631 / 3700


def test_summarize_empty_is_zero_not_division_error():
    s = C.summarize([])
    assert s["turns"] == 0 and s["hit_rate"] == 0.0 and s["warm_hit_rate"] == 0.0


def test_format_report_shows_rates_and_counts():
    report = C.format_report(C.summarize(C.parse_generation_lines(_LOG)))
    assert "turns: 3" in report
    assert "51.9%" in report  # 3631/7000 overall
    assert "98.1%" in report  # 3631/3700 warm-only
