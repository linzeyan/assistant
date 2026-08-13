"""Product-native subagents (N105): parallel fan-out onto the local model.

The dogfood sessions showed users asking the assistant itself to "開 subagent" — and
five different models each improvising (serial edits, ad-hoc scripts, bash child
processes) because no such tool existed. This runner turns one ``spawn_subagents``
tool call into N concurrent child ``AgentLoop`` runs: each child gets a fresh
in-memory ``Session`` (context isolation — the main conversation never sees the
children's tool traffic, only their final summaries), the standard toolset minus
``spawn_subagents`` itself (the schema gate on ``ctx.subagents`` prevents
recursion), and an auto-allow approver — the human approved the fan-out itself,
task list in hand, so children don't re-prompt per tool; S5 deny rules still bind.

Concurrency comes from the N104 batch lane: every child request is marked
``concurrent=True``, so children decode TOGETHER on one model when its engine is
batchable, and degrade to queueing (still correct, just serial) when it isn't.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace

from assistant.tools.approval import PolicyApprover, Rule
from assistant.tools.base import ToolContext
from assistant.tools.registry import ToolRegistry

from .loop import AgentLoop
from .session import Session
from .tokens import cut_at_tokens, estimate_tokens

log = logging.getLogger("assistant")

# Fan-out bounds — backstop constants, not knobs. Each concurrent child holds its own KV
# cache in unified memory, so the task cap is a memory bound as much as a sanity one; the
# iteration cap is tighter than the main loop's because a subagent owns one scoped task,
# not a whole conversation. Result cap keeps N children from blowing the parent's context
# (the loop's N94 batch fuse would truncate anyway — this just cuts per-child, fairly).
_MAX_TASKS = 4
_MAX_ITERS = 6
_RESULT_TOKEN_CAP = 1500


class _LaneLLM:
    """LLM adapter that marks every child request ``concurrent`` so it joins the batch
    lane (N104) instead of queueing behind its sibling subagents."""

    def __init__(self, inner):
        self._inner = inner

    def stream_chat(self, messages, model, tools=None, **params):
        return self._inner.stream_chat(messages, model, tools=tools, concurrent=True, **params)


class SubagentRunner:
    """Owns the child-loop wiring for one backend process; ``run`` executes one fan-out.

    Constructed once at startup (main.py) with the same llm/registry/rules the main loop
    uses, and carried on the shared ``ToolContext`` — child contexts get ``subagents=None``,
    which both hides the tool from their schema and refuses a nested call at runtime.
    """

    def __init__(
        self,
        llm,
        registry: ToolRegistry,
        *,
        approval_rules: list[Rule] | None = None,
        max_output_tokens: int = 4096,
        turn_timeout_s: float | None = None,
    ):
        self._llm = _LaneLLM(llm)
        self._registry = registry
        self._rules = approval_rules or []
        self._max_output_tokens = max_output_tokens
        self._turn_timeout_s = turn_timeout_s

    async def run(self, tasks: object, ctx: ToolContext) -> tuple[bool, str]:
        """Run ``tasks`` as parallel subagents; returns (any_succeeded, combined report)."""
        if (
            not isinstance(tasks, list)
            or not tasks
            or not all(isinstance(t, str) and t.strip() for t in tasks)
        ):
            return False, "spawn_subagents needs 'tasks': a non-empty list of task strings"
        if len(tasks) > _MAX_TASKS:
            return (
                False,
                f"too many tasks ({len(tasks)}); max {_MAX_TASKS} — merge related tasks "
                "or run another batch afterwards",
            )
        if not ctx.model:
            return False, "no model bound to this turn — cannot spawn subagents"

        # One child loop serves all tasks: run() keeps its state per-invocation, so
        # concurrent runs don't interfere, and the children share the ask-once/rule setup.
        child_ctx = replace(ctx, subagents=None, on_progress=None)
        loop = AgentLoop(
            self._llm,
            self._registry,
            PolicyApprover(approval_required=False),  # fan-out itself was the approval
            child_ctx,
            max_iters=_MAX_ITERS,
            max_output_tokens=self._max_output_tokens,
            turn_timeout_s=self._turn_timeout_s,
            approval_rules=self._rules,  # deny rules still bind inside children
        )

        total = len(tasks)
        done = 0

        async def one(index: int, task: str) -> tuple[bool, str]:
            nonlocal done
            try:
                ok, text = await self._run_child(loop, task, ctx.model)
            except Exception as exc:  # one child's crash must not sink its siblings
                log.exception("subagent %d/%d failed", index + 1, total)
                ok, text = False, f"[failed: {exc}]"
            done += 1
            if ctx.on_progress:  # surfaces as tool_progress in the GUI/Telegram
                ctx.on_progress(done / total, f"{done}/{total} subagents done")
            return ok, text

        log.info("subagents: fanning out %d task(s) on %s", total, ctx.model)
        results = await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))

        parts: list[str] = []
        for i, (task, (ok, text)) in enumerate(zip(tasks, results), start=1):
            head = " ".join(task.split())[:80]
            if estimate_tokens(text) > _RESULT_TOKEN_CAP:
                text = cut_at_tokens(text, _RESULT_TOKEN_CAP) + "\n...[subagent result truncated]"
            parts.append(f"### Subagent {i} — {head}\n{text}")
        return any(ok for ok, _ in results), "\n\n".join(parts)

    async def _run_child(self, loop: AgentLoop, task: str, model: str) -> tuple[bool, str]:
        """One subagent turn: fresh session, the task as its only user message. The final
        assistant message is the deliverable; deltas/tool events are deliberately dropped —
        context isolation is the point (the parent pays only for the summary)."""
        session = Session(id=f"sub_{uuid.uuid4().hex[:12]}", model=model)
        error: str | None = None
        async for ev in loop.run(session, task, model):
            if ev["type"] == "error":
                error = ev.get("detail") or "unknown error"
        if error is not None:
            return False, f"[failed: {error}]"
        for m in reversed(session.messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"]:
                return True, m["content"]
        return False, "[failed: subagent produced no answer]"
