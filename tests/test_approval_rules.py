"""Wildcard permission rules (S5): allow/deny/ask over (tool, resource), first-match-wins,
plus ask-once-per-session. These encode WHY — a pre-authorised safe tool must not prompt, a
denied dangerous one must not run, and an already-granted action must not nag again."""

import pytest

from assistant.agent.loop import AgentLoop
from assistant.tools.approval import Rule, resource_of
from assistant.tools.base import Tool, ToolResult


def _tool(name: str, needs_approval: bool = True) -> Tool:
    async def handler(args, ctx):
        return ToolResult(True, "ok")

    return Tool(name=name, description="", parameters={}, handler=handler, needs_approval=needs_approval)


def _loop(rules=None, ask_once: bool = True) -> AgentLoop:
    # _rule_decision/_remember touch only rule state, so the other deps can be None.
    return AgentLoop(None, None, None, None, approval_rules=rules, approval_ask_once=ask_once)


class _FakeRegistry:
    def __init__(self, names):
        self._names = names

    def schemas(self):
        return [{"type": "function", "function": {"name": n}} for n in self._names]


def _loop_with_tools(names, rules=None) -> AgentLoop:
    return AgentLoop(None, _FakeRegistry(names), None, None, approval_rules=rules)


def _visible_names(loop) -> list[str]:
    return [s["function"]["name"] for s in (loop._visible_tool_schemas() or [])]


# --- Rule parsing / matching ---
def test_rule_from_dict_rejects_bad_decision():
    with pytest.raises(ValueError):
        Rule.from_dict({"action": "bash", "decision": "maybe"})


def test_rule_from_dict_requires_action():
    with pytest.raises(ValueError):
        Rule.from_dict({"decision": "allow"})


def test_rule_matches_action_and_resource_globs():
    r = Rule(action="write_*", resource="src/*", decision="deny")
    assert r.matches("write_file", "src/main.py")
    assert not r.matches("read_file", "src/main.py")  # action glob misses
    assert not r.matches("write_file", "docs/readme.md")  # resource glob misses


def test_resource_of_precedence_and_empty():
    assert resource_of({"path": "a", "command": "b"}) == "a"  # path beats command
    assert resource_of({"command": "ls"}) == "ls"
    assert resource_of({}) == ""


# --- loop decision ---
def test_safe_tool_always_allows():
    assert _loop()._rule_decision(_tool("read_file", needs_approval=False), {}) == "allow"


def test_allow_rule_skips_prompt():
    loop = _loop([Rule("bash", "*", "allow")])
    assert loop._rule_decision(_tool("bash"), {"command": "ls"}) == "allow"


def test_deny_rule_refuses():
    loop = _loop([Rule("bash", "rm *", "deny")])
    assert loop._rule_decision(_tool("bash"), {"command": "rm -rf x"}) == "deny"


def test_no_rule_falls_through_to_ask():
    assert _loop()._rule_decision(_tool("bash"), {"command": "ls"}) == "ask"


def test_deny_wins_regardless_of_position():
    # deny-priority: even with allow listed last, a matching deny must win (fail safe).
    loop = _loop([Rule("bash", "rm *", "deny"), Rule("bash", "rm *", "allow")])
    assert loop._rule_decision(_tool("bash"), {"command": "rm x"}) == "deny"


def test_last_match_wins_among_non_deny():
    loop = _loop([Rule("bash", "*", "allow"), Rule("bash", "make *", "ask")])
    assert loop._rule_decision(_tool("bash"), {"command": "make test"}) == "ask"  # last match
    assert loop._rule_decision(_tool("bash"), {"command": "ls"}) == "allow"  # only the first


def test_ask_once_remembers_exact_resource_only():
    loop = _loop()  # no rules -> default ask
    tool, args = _tool("bash"), {"command": "make test"}
    assert loop._rule_decision(tool, args) == "ask"
    loop._remember(tool, args)
    assert loop._rule_decision(tool, args) == "allow"  # same command -> no re-prompt
    assert loop._rule_decision(tool, {"command": "make build"}) == "ask"  # different -> ask


def test_ask_once_disabled_never_remembers():
    loop = _loop(ask_once=False)
    tool, args = _tool("bash"), {"command": "make test"}
    loop._remember(tool, args)
    assert loop._rule_decision(tool, args) == "ask"


# --- schema filtering (S5): blanket deny hides the tool from the model ---
def test_blanket_deny_hides_tool_from_schema():
    loop = _loop_with_tools(["bash", "read_file"], [Rule("bash", "*", "deny")])
    assert _visible_names(loop) == ["read_file"]  # bash never offered


def test_resource_specific_deny_keeps_tool_visible():
    # Denying only `rm *` doesn't make bash unusable — it stays in the schema, enforced live.
    loop = _loop_with_tools(["bash", "read_file"], [Rule("bash", "rm *", "deny")])
    assert _visible_names(loop) == ["bash", "read_file"]


def test_no_rules_shows_all_tools():
    loop = _loop_with_tools(["bash", "read_file"])
    assert _visible_names(loop) == ["bash", "read_file"]


def test_is_blanket_deny_predicate():
    assert Rule("bash", "*", "deny").is_blanket_deny("bash")
    assert Rule("*", "**", "deny").is_blanket_deny("write_file")  # wildcard action + all
    assert not Rule("bash", "rm *", "deny").is_blanket_deny("bash")  # resource-specific
    assert not Rule("bash", "*", "allow").is_blanket_deny("bash")  # not a deny
