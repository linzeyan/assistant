"""The update_plan tool (SA.3): normalize_steps validation + the tool's short-ack contract."""

from __future__ import annotations

import pytest

from assistant.tools.base import ToolContext
from assistant.tools.plan_tool import normalize_steps, plan_summary, update_plan


def test_normalize_drops_empty_titles_and_defaults_bad_status():
    out = normalize_steps([
        {"title": "  read  ", "status": "in_progress"},
        {"title": "", "status": "pending"},          # dropped
        {"title": "ship", "status": "DONE-ish"},      # unknown status → pending
        {"title": "fix", "status": "completed"},
    ])
    assert out == [
        {"title": "read", "status": "in_progress"},
        {"title": "ship", "status": "pending"},
        {"title": "fix", "status": "completed"},
    ]


def test_normalize_rejects_unusable_input():
    for bad in (None, [], "steps", [{"status": "pending"}], [{"title": "   "}]):
        with pytest.raises(ValueError):
            normalize_steps(bad)


def test_plan_summary_counts_completed():
    assert plan_summary([
        {"title": "a", "status": "completed"},
        {"title": "b", "status": "pending"},
    ]) == "1/2 done"


async def test_update_plan_acks_briefly_without_leaking_titles(tmp_path):
    # The tool result is what history sees — it must stay short (no full checklist), since the
    # loop owns the authoritative plan and emits it as an event instead (SA.3 必補#1).
    res = await update_plan(
        {"steps": [{"title": "secret step title", "status": "pending"}]},
        ToolContext(cwd=tmp_path),
    )
    assert res.ok
    assert "plan updated" in res.content
    assert "secret step title" not in res.content


async def test_update_plan_fails_loud_on_bad_steps(tmp_path):
    res = await update_plan({"steps": []}, ToolContext(cwd=tmp_path))
    assert not res.ok
    assert "invalid plan" in res.content
