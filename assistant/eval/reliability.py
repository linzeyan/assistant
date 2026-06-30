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
NO_CALL = "no_call"  # death mode 1
PARSE_MISS = "parse_miss"  # death mode 2
TOOL_ERROR = "tool_error"  # death mode 3
CRASH = "crash"  # turn died mid-loop (e.g. chat-template render TypeError) — a mode-3 upstream variant
MAX_ITERS = "max_iters"  # loop control: ran out of tool steps
TIMEOUT = "timeout"  # loop control: exceeded the per-turn wall-clock budget (B1)
THRASH = "thrash"  # loop control: repeated an identical tool call with no progress (B2)

# Order is the report's display order: success first, then the four death modes, then loop control.
BUCKETS = (SUCCESS, NO_CALL, PARSE_MISS, TOOL_ERROR, CRASH, MAX_ITERS, TIMEOUT, THRASH)

LABELS = {
    SUCCESS: "success (answered, tool ran)",
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


def classify_turn(trace: dict, *, expects_tool: bool = True) -> str:
    """Bucket one turn trace. ``expects_tool`` says whether the prompt was meant to require a tool;
    when False, an ``answered`` turn with no tool call is a legitimate success, not death mode 1.

    A non-``answered`` outcome already encodes the failure (the loop's ``finalize`` classified it):
    ``parse_miss`` / ``tool_error`` map to modes 2 / 3, ``error`` to a mid-loop crash, and
    ``max_iters`` / ``timeout`` to loop control. Only a clean ``answered`` needs the extra
    no-call-vs-success split, which mode 4 (ignored result) hides behind — see the module docstring.
    """
    outcome = trace.get("outcome")
    if outcome in (PARSE_MISS, TOOL_ERROR, MAX_ITERS, TIMEOUT, THRASH):
        return outcome
    if outcome == "error":
        return CRASH
    # outcome == "answered" (or anything unexpected): success unless a tool was expected but never
    # attempted (death mode 1).
    if expects_tool and not _called_a_tool(trace):
        return NO_CALL
    return SUCCESS


def summarize(buckets: list[str]) -> dict:
    """Aggregate per-turn buckets into a report: total, success count, success rate, breakdown."""
    counts = Counter(buckets)
    total = len(buckets)
    success = counts.get(SUCCESS, 0)
    return {
        "total": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
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
    lines = [
        head,
        "─" * len(head),
        f"runs: {total}" + (f" (×{runs} per prompt)" if runs else ""),
        f"success: {summary['success']}/{total}  ({rate:.0f}%)",
        "",
        "failures by death mode:" if summary["breakdown"] else "no failures recorded.",
    ]
    for bucket, count in summary["breakdown"]:
        if bucket == SUCCESS:
            continue
        lines.append(f"  {count:>3}  {LABELS.get(bucket, bucket)}")
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
# Caveat on the numbers: ``tool_error`` counts a turn where *any* tool call failed, even one the
# model then recovered from (e.g. a fetch_url 403 it routed around) — the coarse trace outcome can't
# tell a fatal failure from a recovered one, so it slightly *under*-counts real success. Honest
# floor, not the ceiling.
DEFAULT_PROMPTS = (
    "Search the web for the latest stable version of Python and tell me the version number.",
    "Use a web search to find who currently holds the men's 100m world record, and the time.",
    "Search the web for the current USD to JPY exchange rate and state the rate.",
    "Find, with a web search, what the latest released version of the Rust language is.",
)
