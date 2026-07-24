"""Tool-calling reliability measurement (spring5 A1) — the *measure* in measure-before-fix.

The agent loop already records a per-turn trace (spring2 P0) with a coarse ``outcome``. This module
turns a batch of those traces into **numbers**: a success rate, and every failure bucketed into one
of the four ways tool-calling dies (spring2 §3), whose fixes are completely different:

  1. ``no_call``    — the model never emitted a tool call (a decision / model-quality miss)
  2. ``parse_miss`` — it emitted one but the parser missed it (it leaked back as visible text)
  3. ``tool_error`` — the call ran but the tool backend failed (no network / load failure / …)
  4. ``ignored``    — the tool returned but the model ignored the result and answered anyway

Modes 1–3 are read straight from the trace (``outcome`` + steps). **Mode 4 is deliberately NOT
auto-classified**: deciding the model "ignored" a good tool result requires judging answer quality,
which a trace can't tell us — an ``answered`` turn that did call a tool is counted as ``success``
here and flagged for human / LLM-judge review. Reporting a fake mode-4 number would be worse than
admitting we can't see it.

This module is pure — it consumes trace dicts (the shape of ``TurnTrace.to_dict`` / the
``GET /turns/{id}`` body). The live driver that produces those traces by exercising a running
backend lives in ``assistant.eval.measure`` (``python -m assistant.eval.measure``).
"""

from __future__ import annotations

from collections import Counter

# Buckets a turn can land in. The four death modes plus the loop-control outcomes (which are real
# failures too — a turn that hit the iteration ceiling or the wall-clock budget didn't succeed).
SUCCESS = "success"
# Answered, and a tool did run, but some other call failed along the way and the model routed
# around it. Structurally a delivered turn — same epistemic standing as SUCCESS (neither judges
# answer quality; see mode 4) — but tracked apart so the flailing stays visible.
RECOVERED = "recovered"
NO_CALL = "no_call"  # death mode 1
PARSE_MISS = "parse_miss"  # death mode 2
TOOL_ERROR = "tool_error"  # death mode 3
CRASH = "crash"  # turn died mid-loop (e.g. chat-template render TypeError) — a mode-3 upstream variant
MAX_ITERS = "max_iters"  # loop control: ran out of tool steps
TIMEOUT = "timeout"  # loop control: exceeded the per-turn wall-clock budget (B1)
THRASH = "thrash"  # loop control: repeated an identical tool call with no progress (B2)

# Order is the report's display order: delivered first, then the four death modes, then loop control.
BUCKETS = (SUCCESS, RECOVERED, NO_CALL, PARSE_MISS, TOOL_ERROR, CRASH, MAX_ITERS, TIMEOUT, THRASH)
# Buckets that mean the turn delivered an answer with a tool in the loop.
DELIVERED = (SUCCESS, RECOVERED)

LABELS = {
    SUCCESS: "success (answered, tool ran)",
    RECOVERED: "recovered (answered after a tool call failed)",
    NO_CALL: "mode 1 · no tool call emitted",
    PARSE_MISS: "mode 2 · emitted but parser missed it",
    TOOL_ERROR: "mode 3 · tool backend failed",
    CRASH: "mode 3* · turn crashed mid-loop",
    MAX_ITERS: "loop · hit the tool-step ceiling",
    TIMEOUT: "loop · hit the turn time limit",
    THRASH: "loop · stuck repeating a tool call",
}


def _called_a_tool(trace: dict) -> bool:
    """True if any step parsed at least one tool call — i.e. the model actually attempted a tool,
    distinguishing a genuine ``success`` from death mode 1 (answered without ever calling one)."""
    return any(step.get("parsed_calls") for step in trace.get("steps") or [])


def _any_tool_succeeded(trace: dict) -> bool:
    """True if at least one tool call returned ok — i.e. the model got real data to answer from,
    even if another call failed."""
    return any(
        r.get("ok")
        for step in trace.get("steps") or []
        for r in step.get("tool_results") or []
    )


def classify_turn(trace: dict, *, expects_tool: bool = True) -> str:
    """Bucket one turn trace. ``expects_tool`` says whether the prompt was meant to require a tool;
    when False, an ``answered`` turn with no tool call is a legitimate success, not death mode 1.

    A non-``answered`` outcome already encodes the failure (the loop's ``finalize`` classified it):
    ``parse_miss`` / ``tool_error`` map to modes 2 / 3, ``error`` to a mid-loop crash, and
    ``max_iters`` / ``timeout`` to loop control. Only a clean ``answered`` needs the extra
    no-call-vs-success split, which mode 4 (ignored result) hides behind — see the module docstring.
    """
    outcome = trace.get("outcome")
    if outcome == TOOL_ERROR:
        # ``finalize`` only marks tool_error on a turn that reached an answer, so the trace's own
        # outcome can't say whether the failure was terminal. It is if NOTHING worked — the model
        # answered from nothing. If some call did return data, the model routed around the failure,
        # which is a delivered turn, not death mode 3. Counting those as failures is what made a
        # 15/16 sweep report 38%.
        return RECOVERED if _any_tool_succeeded(trace) else TOOL_ERROR
    if outcome in (PARSE_MISS, MAX_ITERS, TIMEOUT, THRASH):
        return outcome
    if outcome == "error":
        return CRASH
    # outcome == "answered" (or anything unexpected): success unless a tool was expected but never
    # attempted (death mode 1).
    if expects_tool and not _called_a_tool(trace):
        return NO_CALL
    return SUCCESS


def summarize(buckets: list[str]) -> dict:
    """Aggregate per-turn buckets into a report: total, delivered count, rate, breakdown.

    The headline rate is over DELIVERED (clean + recovered), because that's what a user
    experiences: the turn answered with a tool in the loop. ``success`` keeps the strict count
    so a rise in ``recovered`` — the model flailing but coping — stays visible."""
    counts = Counter(buckets)
    total = len(buckets)
    success = counts.get(SUCCESS, 0)
    delivered = sum(counts.get(b, 0) for b in DELIVERED)
    return {
        "total": total,
        "success": success,
        "recovered": counts.get(RECOVERED, 0),
        "delivered": delivered,
        "success_rate": (delivered / total) if total else 0.0,
        # Stable, display-ordered breakdown (omit buckets that never occurred to keep it scannable).
        "breakdown": [(b, counts[b]) for b in BUCKETS if counts.get(b)],
    }


def format_report(summary: dict, *, model: str | None = None, runs: int | None = None) -> str:
    """Render ``summarize``'s output as a compact human table. Pure string — the CLI prints it."""
    total = summary["total"]
    rate = summary["success_rate"] * 100
    head = "tool-calling reliability"
    if model:
        head += f" · {model}"
    delivered = summary.get("delivered", summary["success"])
    headline = f"delivered: {delivered}/{total}  ({rate:.0f}%)"
    if summary.get("recovered"):
        # Spell out the split: a high rate carried by recoveries is a different health story
        # from a clean one, and the difference must not hide inside one percentage.
        headline += f"  — {summary['success']} clean, {summary['recovered']} after a failed call"
    failures = [(b, c) for b, c in summary["breakdown"] if b not in DELIVERED]
    lines = [
        head,
        "─" * len(head),
        f"runs: {total}" + (f" (×{runs} per prompt)" if runs else ""),
        headline,
        "",
        "failures by death mode:" if failures else "no failures recorded.",
    ]
    for bucket, count in failures:
        lines.append(f"  {count:>3}  {LABELS.get(bucket, bucket)}")
    return "\n".join(lines)


# Per-prompt transition matrix between two capture sets (the rows ``measure --append-corpus``
# writes). Aggregate rates hide compensating changes — a fix that repairs prompt A while breaking
# prompt B can leave the headline number unchanged; per-prompt transitions can't. A prompt "passes"
# a sweep when at least half its runs delivered (stochastic models make a single run too noisy to
# be a verdict; the raw x/n counts stay in the output so the margin is visible either way).

FAIL_TO_PASS = "fail_to_pass"
PASS_TO_FAIL = "pass_to_fail"  # the regression signal — surfaced first, loudly
IMPROVED = "improved"  # same side of the pass line, delivered rate went up
REGRESSED = "regressed"  # same side, rate went down
UNCHANGED = "unchanged"
BASELINE_ONLY = "baseline_only"  # prompt absent from one side — compared sweeps differ
CANDIDATE_ONLY = "candidate_only"

_TRANSITION_ORDER = (
    PASS_TO_FAIL, REGRESSED, FAIL_TO_PASS, IMPROVED, UNCHANGED, BASELINE_ONLY, CANDIDATE_ONLY,
)


def _delivered_by_prompt(rows: list[dict], *, expects_tool: bool) -> dict[str, tuple[int, int]]:
    """prompt text -> (delivered, total) over capture rows (which carry the same outcome/steps
    fields ``classify_turn`` reads)."""
    stats: dict[str, tuple[int, int]] = {}
    for row in rows:
        prompt = row.get("user_text") or "(unknown prompt)"
        d, t = stats.get(prompt, (0, 0))
        bucket = classify_turn(row, expects_tool=expects_tool)
        stats[prompt] = (d + (1 if bucket in DELIVERED else 0), t + 1)
    return stats


def compare_by_prompt(
    baseline: list[dict], candidate: list[dict], *, expects_tool: bool = True
) -> list[dict]:
    """Classify every prompt's transition from a baseline sweep to a candidate sweep.

    Returns display-ordered rows ``{prompt, baseline: (d, n) | None, candidate: (d, n) | None,
    transition}`` with regressions first."""
    base = _delivered_by_prompt(baseline, expects_tool=expects_tool)
    cand = _delivered_by_prompt(candidate, expects_tool=expects_tool)
    rows = []
    for prompt in base.keys() | cand.keys():
        b, c = base.get(prompt), cand.get(prompt)
        if b is None:
            transition = CANDIDATE_ONLY
        elif c is None:
            transition = BASELINE_ONLY
        else:
            b_rate, c_rate = b[0] / b[1], c[0] / c[1]
            b_pass, c_pass = b_rate >= 0.5, c_rate >= 0.5
            if b_pass and not c_pass:
                transition = PASS_TO_FAIL
            elif not b_pass and c_pass:
                transition = FAIL_TO_PASS
            elif c_rate > b_rate:
                transition = IMPROVED
            elif c_rate < b_rate:
                transition = REGRESSED
            else:
                transition = UNCHANGED
        rows.append(
            {"prompt": prompt, "baseline": b, "candidate": c, "transition": transition}
        )
    rows.sort(key=lambda r: (_TRANSITION_ORDER.index(r["transition"]), r["prompt"]))
    return rows


def format_comparison(rows: list[dict]) -> str:
    """Render ``compare_by_prompt`` as a compact table, regressions on top."""
    if not rows:
        return "no prompts to compare."
    head = "per-prompt transitions (baseline → candidate)"
    lines = [head, "─" * len(head)]
    for r in rows:
        b, c = r["baseline"], r["candidate"]
        left = f"{b[0]}/{b[1]}" if b else "—"
        right = f"{c[0]}/{c[1]}" if c else "—"
        prompt = r["prompt"] if len(r["prompt"]) <= 60 else r["prompt"][:57] + "…"
        lines.append(f"  {r['transition']:<14} {left:>5} → {right:<5}  {prompt}")
    regressions = sum(r["transition"] in (PASS_TO_FAIL, REGRESSED) for r in rows)
    lines.append("")
    lines.append(
        f"⚠ {regressions} prompt(s) regressed — inspect before trusting the headline rate."
        if regressions
        else "no per-prompt regressions."
    )
    return "\n".join(lines)


# Default probe prompts: each is meant to *require* a tool, so an ``answered`` turn that never called
# one is a real death-mode-1 miss. These exercise the web tools (web_search / fetch_url), which work
# API-side without any workspace. Filesystem/coding prompts are intentionally NOT here: an
# API-driven chat has no workspace (it defaults to the backend's cwd, e.g. ``/``) and ``bash``
# requires approval, so file prompts measure the environment, not tool-calling — run those via
# --prompts against a workspace-configured backend (the GUI, or Telegram after /cd). Example coding
# set for that case:
#   "List the files in the current working directory."
#   "Read this project's README and summarise it in two sentences."
#
# Caveat on the numbers: ``recovered`` vs ``tool_error`` is a *structural* split (did any tool
# return data before the model answered), not an answer-quality judgment — a turn that routed
# around a failure and then answered wrongly still counts as delivered, exactly as a clean
# ``success`` with a wrong answer does. Mode 4 stays invisible to both; see the module docstring.
DEFAULT_PROMPTS = (
    "Search the web for the latest stable version of Python and tell me the version number.",
    "Use a web search to find who currently holds the men's 100m world record, and the time.",
    "Search the web for the current USD to JPY exchange rate and state the rate.",
    "Find, with a web search, what the latest released version of the Rust language is.",
)
