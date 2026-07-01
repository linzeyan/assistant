"""System-prompt assembly + send-time reference blocks (date / memory).

The date block fixes local models hallucinating "today" from their training cutoff. It must
ride the user turn, NEVER the cacheable system prefix (S3) — pinned here."""

from __future__ import annotations

from datetime import datetime, timezone

from assistant.agent.prompt import (
    build_system_prompt,
    wrap_datetime_context,
    wrap_memory_context,
    wrap_referenced_paths,
)


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


def test_referenced_paths_block_is_reference_only_and_lists_paths():
    block = wrap_referenced_paths(["/a/Makefile", "/b/README.md"])
    assert block.startswith("<referenced-paths reference-only>")
    assert "/a/Makefile" in block and "/b/README.md" in block
    assert "read_file" in block and "BEFORE" in block
