"""System-prompt assembly + send-time reference blocks (date / memory).

The date block fixes local models hallucinating "today" from their training cutoff. It must
ride the user turn, NEVER the cacheable system prefix (S3) — pinned here."""

from __future__ import annotations

from datetime import datetime, timezone

from assistant.agent.prompt import (
    build_system_prompt,
    wrap_datetime_context,
    wrap_memory_context,
    wrap_plan_context,
    wrap_referenced_paths,
)
from assistant.agent.tokens import estimate_tokens


def test_datetime_context_includes_date_and_weekday():
    now = datetime(2026, 6, 26, 14, 30, tzinfo=timezone.utc)
    block = wrap_datetime_context(now)
    assert "2026-06-26" in block
    assert now.strftime("%A") in block  # weekday, so the model can reason about recency
    assert block.startswith("<current-datetime")


def test_system_prompt_has_no_date():
    # The date must NOT live in the system prompt — that would break the byte-stable,
    # cacheable prefix (a fresh fingerprint every minute → KV-cache miss every turn).
    p = build_system_prompt("(skills index)")
    assert "current-datetime" not in p
    assert "2026" not in p


def test_memory_context_is_reference_only_block():
    assert wrap_memory_context("a fact").startswith("<memory-context reference-only>")


def test_system_prompt_has_skill_use_policy():
    # SB.2: a static policy nudging the model to skill_view a clearly-matching skill rather than
    # improvise. Lives in the cacheable prefix (not per-turn), so it must be in build_system_prompt
    # regardless of the skills index passed.
    p = build_system_prompt("(skills index)")
    assert "Using skills" in p
    assert "skill_view" in p


def test_system_prompt_has_plan_and_ethos():
    # SA.3: static guidance to use the update_plan checklist for multi-step tasks plus the ETHOS
    # principles (investigate-first / finish / let the user decide). Both are static, so they
    # belong in the cacheable prefix.
    p = build_system_prompt("(skills index)")
    assert "update_plan" in p
    assert "multi-step" in p


def test_system_prompt_has_grounding_policy():
    # N53: a weak-at-tools model answered "看下 <Makefile>" from imagination — never opened the
    # file, fabricated commands (`brew install mlx`) and a made-up code API. The static grounding
    # rules (read named files first / don't fabricate / answer the actual question) live in the
    # cacheable prefix, so they must be present regardless of the skills index.
    p = build_system_prompt("(skills index)")
    assert "READ it with a tool" in p  # read a named file before answering
    assert "Do not fabricate" in p  # no invented commands / APIs
    assert "fetch_url" in p  # verify from the real source


def test_system_prompt_treats_a_failed_read_as_a_stop():
    # Observed: asked to edit a file that was not in the worktree (a gitignored symlink), the
    # model's read_file returned "not a file", it marked its own "read the file" step completed,
    # and then wrote the file from imagination — a plausible-looking document that had never
    # existed. The grounding rules above only covered *not reading*; a read that failed still
    # looked to the model like a read. Static policy, so it belongs in the cacheable prefix.
    p = build_system_prompt("(skills index)")
    assert "A tool call that FAILED opened nothing" in p
    assert "report that mismatch and stop" in p


def test_system_prompt_reports_the_change_actually_made():
    # Observed: asked for three edits to a Makefile, the model made two, then reported all three
    # ("Fixed the wait loop…") — its own diff contradicted the claim. In the same turn a bash call
    # died on the 120s timeout and the run was reported as tests passing. The transcript rule above
    # only covered inventing OUTPUT; this covers inventing one's own EDITS and treating a killed
    # command as a result. Static policy, so it belongs in the cacheable prefix.
    p = build_system_prompt("(skills index)")
    assert "Report the change you MADE" in p
    assert "cut short" in p and "has established nothing" in p


def test_system_prompt_has_code_mode_batch_policy():
    # N100: for fan-out/loop work the model should write ONE script (bash tool: shell or
    # python3 heredoc) instead of N sequential tool calls — intermediate results stay in the
    # subprocess and only the distilled output enters context (the "code mode" token mechanic;
    # also fewer round trips for weak-at-tools local models). Static policy, so it must live
    # in the cacheable prefix regardless of the skills index.
    p = build_system_prompt("(skills index)")
    assert "ONE script" in p
    assert "distilled" in p
    assert "heredoc" in p  # python3 inline via bash is the sanctioned escape hatch


def test_skills_index_is_budget_capped():
    # N102: skill_manage grows the library unboundedly — the catalog must not slowly eat the
    # prompt. Over budget we keep a deterministic prefix (stable across turns, so the cacheable
    # prefix survives) and point at skills_list for the rest; a small index passes untouched.
    huge = "\n".join(f"- skill-{i:04d}: does thing number {i}" for i in range(2000))
    p = build_system_prompt(huge)
    assert "index truncated" in p and "skills_list" in p
    # Raw index alone is ~17k tokens, so any ceiling in this range proves the cut happened.
    # Kept well clear of the current size: a bound that tracks the base prompt turns every
    # added line of static policy into a failure of a test about the *index*.
    assert estimate_tokens(p) < 4500

    small = build_system_prompt("- one: does a thing")
    assert "- one: does a thing" in small and "index truncated" not in small


def test_plan_context_block_lists_steps_and_is_bounded():
    # N102: an unfinished plan rides the next user turn (never the S3 prefix) so the model
    # keeps its own checklist across turns; a runaway list must not flood the turn.
    steps = [{"title": f"step {i}", "status": "pending"} for i in range(40)]
    block = wrap_plan_context(steps)
    assert block.startswith("<plan reference-only>")
    assert "- [pending] step 0" in block and "step 29" in block
    assert "step 30" not in block and "+10 more" in block
    assert "update_plan" in block  # tells the model how to continue/revise it


def test_referenced_paths_block_is_reference_only_and_lists_paths():
    block = wrap_referenced_paths(["/a/Makefile", "/b/README.md"])
    assert block.startswith("<referenced-paths reference-only>")
    assert "/a/Makefile" in block and "/b/README.md" in block
    assert "read_file" in block and "BEFORE" in block
