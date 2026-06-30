from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from assistant.tools.approval import ApprovalPolicy, Rule, resource_of
from assistant.tools.base import ToolContext, ToolResult
from assistant.tools.plan_tool import normalize_steps
from assistant.tools.registry import ToolRegistry

from .compaction import CompactionManager
from .diff import build_turn_changes
from .git_changes import dirty_paths, head_bytes, repo_root
from .hooks import HookRegistry
from .llm_client import AsyncLLM
from .prompt import (
    build_system_prompt,
    wrap_datetime_context,
    wrap_memory_context,
    wrap_workspace_context,
)
from .session import Session
from .tokens import estimate_messages_tokens, estimate_tokens
from .trace import TraceStep, TraceStore, TurnTrace

log = logging.getLogger("assistant")


class AgentLoop:
    """The think -> tool -> observe cycle.

    Each iteration streams one assistant turn. Text deltas are forwarded to the
    caller; if the model emits tool calls, each is approval-checked, dispatched, its
    result appended to the conversation, and the loop runs again. When a turn
    produces no tool calls, that turn is the final answer.

    `run()` yields typed event dicts (assistant_delta / tool_call / tool_result /
    done / error) which the API layer forwards verbatim as SSE.
    """

    def __init__(
        self,
        llm: AsyncLLM,
        registry: ToolRegistry,
        approver: ApprovalPolicy,
        ctx: ToolContext,
        max_iters: int = 8,
        compaction: CompactionManager | None = None,
        max_output_tokens: int = 4096,
        approval_rules: list[Rule] | None = None,
        approval_ask_once: bool = True,
        hooks: HookRegistry | None = None,
        trace_store: TraceStore | None = None,
    ):
        self._llm = llm
        self._registry = registry
        self._approver = approver
        self._ctx = ctx
        self._max_iters = max_iters
        self._compaction = compaction
        self._max_output_tokens = max_output_tokens
        # Per-turn trace (spring2 P0): None = off. Recording is a side-channel — it never
        # changes the events yielded to the caller.
        self._trace = trace_store
        # In-process hook seam (P2); empty registry by default so every fire is a no-op.
        self._hooks = hooks or HookRegistry()
        # Wildcard permission rules (S5), layered over whichever approver a run uses. The
        # ask-once grant set is process-scoped (resets on backend restart), so an approved
        # action isn't re-prompted for the life of the backend.
        self._rules = approval_rules or []
        self._ask_once = approval_ask_once
        self._granted: set[tuple[str, str]] = set()

    async def run(
        self,
        session: Session,
        user_text: str,
        model: str,
        approver: ApprovalPolicy | None = None,
        cwd: str | Path | None = None,
    ) -> AsyncIterator[dict]:
        # Per-run approver lets the Telegram gateway inject an interactive (inline
        # button) approver while the desktop path uses the default policy approver.
        effective_approver = approver or self._approver
        # Per-run working directory: workspace is per-conversation, so the Telegram gateway
        # passes the chat's chosen dir here. A copy (not a mutation of the shared ctx) keeps
        # concurrent turns isolated and leaves the desktop/HTTP default untouched.
        ctx = self._ctx if cwd is None else replace(self._ctx, cwd=Path(cwd))
        # Prefix stability (S2+S3): the system prompt is STABLE (base + skills index only)
        # and is reinstalled only when its fingerprint changes, so the cacheable prefix
        # stays byte-identical turn-to-turn. Per-turn memory is kept OUT of it and instead
        # rides the latest user message at send time (see _messages_for_send).
        stable_system = self._build_stable_system()
        fingerprint = self._system_fingerprint(stable_system, model)
        if session.ensure_system(stable_system, fingerprint) == "changed":
            log.warning(
                "system prompt prefix rebuilt (KV-cache miss): fingerprint changed, model=%s",
                model,
            )
        # Compaction (S6) runs on the existing history, before the new turn is added, so a
        # long session is summarized down to make room. Emitted so the UI can note it.
        if self._compaction is not None:
            compaction_event = await self._compaction.maybe_compact(session, model)
            if compaction_event is not None:
                yield compaction_event
        session.add_user(user_text)
        await self._hooks.fire_before_start(session, user_text)  # P2 hook seam
        memory_block = await self._prefetch_memory(user_text)
        tools = self._visible_tool_schemas()
        # Reference-only blocks injected into the latest user message at send time (never the
        # cacheable system prefix, S3). The date is stamped once per turn so a local model —
        # which has no clock — stops hallucinating "today" from its training cutoff.
        context_blocks = [
            wrap_datetime_context(datetime.now().astimezone()),
            wrap_workspace_context(ctx.cwd),
            wrap_memory_context(memory_block) if memory_block else "",
        ]
        # P0 trace: assemble a per-turn record as the loop runs; recorded at each exit point.
        trace = TurnTrace.new(session.id, model, user_text) if self._trace else None
        step: TraceStep | None = None  # current iteration's step; visible to the except below
        # Turn-scoped file snapshots (abs path -> bytes before its first edit this turn) so we
        # can return a diff of what the agent changed (Spring 2 P2/P3). Local, not instance
        # state, so concurrent turns don't clobber each other.
        turn_snapshots: dict[str, bytes | None] = {}
        # git baseline captured on the first bash call, so the turn diff can also show files
        # shell touched (not just write/edit targets). Empty until then; see _git_changes.py.
        shell_state: dict = {}
        # Absolute paths this turn's generate_image/edit_image produced. The agent sometimes
        # follows an image generation with a needless view_image on its own output — loading a
        # heavy vision model to redescribe a picture the user already received. We skip that
        # deterministically below rather than relying on the model not to do it.
        generated_images: set[str] = set()

        try:
            for _ in range(self._max_iters):
                text_parts: list[str] = []
                tool_calls: list[dict] | None = None
                step = TraceStep() if trace is not None else None

                send_messages = self._messages_for_send(session.messages, context_blocks)
                async for ev in self._llm.stream_chat(
                    send_messages, model, tools=tools, max_tokens=self._max_output_tokens
                ):
                    if ev["type"] == "text":
                        text_parts.append(ev["content"])
                        yield {"type": "assistant_delta", "content": ev["content"]}
                    elif ev["type"] == "tool_calls":
                        tool_calls = ev["tool_calls"]
                    else:
                        # Forward any other event the model layer emits (e.g. Fusion's panel
                        # tool_progress) straight to the gateway/SSE consumer.
                        yield ev

                if not tool_calls:
                    answer = "".join(text_parts)
                    session.add_assistant(answer)
                    if trace is not None:
                        step.model_text = answer
                        trace.steps.append(step)
                        trace.final_text = answer
                        self._trace.record(trace.finalize("answered"))
                    # Return what the agent changed on disk so a gateway can show a diff —
                    # code results are first-class, not just the text reply. Fold in any
                    # shell-touched files (via git) before building the diff.
                    self._merge_shell_changes(shell_state, turn_snapshots)
                    diff_event = self._turn_diff_event(turn_snapshots, ctx)
                    if diff_event is not None:
                        yield diff_event
                    # Surface a context-usage readout (estimate): context_tokens is the whole
                    # conversation's footprint — the number compaction (S6) keys off — and
                    # output_tokens is this reply. Heuristic; see agent/tokens.py.
                    yield {
                        "type": "done",
                        "usage": {
                            "context_tokens": estimate_messages_tokens(session.messages),
                            "output_tokens": estimate_tokens(answer),
                        },
                    }
                    return

                session.messages.append(self._assistant_tool_msg(text_parts, tool_calls))
                if trace is not None:
                    step.model_text = "".join(text_parts)
                    step.parsed_calls = [
                        {"name": tc["name"], "arguments": tc["arguments"]} for tc in tool_calls
                    ]
                for tc in tool_calls:
                    yield {
                        "type": "tool_call",
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                    tool = self._registry.get(tc["name"])
                    result: ToolResult | None = None
                    run_it = False  # set by whichever branch authorizes execution
                    # Short-circuit a view_image on an image we just generated this turn (the
                    # user already has it; viewing it only spins up a vision model).
                    _vp = tc["arguments"].get("path") if tc["name"] == "view_image" else None
                    if _vp is not None and (
                        _vp in generated_images or str(ctx.cwd / _vp) in generated_images
                    ):
                        result = ToolResult(
                            True, "(image already generated and shown to the user)"
                        )
                    elif tool is None:
                        result = ToolResult(False, f"unknown tool: {tc['name']}")
                    else:
                        # S5: rules + ask-once decide allow/deny before any prompt; only an
                        # unresolved "ask" falls through to the interactive/policy approver.
                        decision = self._rule_decision(tool, tc["arguments"])
                        if decision == "allow":
                            run_it = True
                        elif decision == "deny":
                            result = ToolResult(False, f"denied by rule: tool '{tool.name}'")
                        elif tool.needs_approval and getattr(
                            effective_approver, "interactive", False
                        ):
                            # Interactive approver (GUI/HTTP): surface the request and wait
                            # for an out-of-band decision (POST /chat/approve) before running.
                            token = effective_approver.new_request()
                            yield {
                                "type": "approval_request",
                                "id": tc["id"],
                                "token": token,
                                "name": tool.name,
                                "arguments": tc["arguments"],
                            }
                            if await effective_approver.wait(token):
                                self._remember(tool, tc["arguments"])  # ask-once-per-session
                                run_it = True
                            else:
                                result = ToolResult(
                                    False, f"denied: tool '{tool.name}' requires approval"
                                )
                        elif not await effective_approver.approve(tool, tc["arguments"]):
                            result = ToolResult(
                                False, f"denied: tool '{tool.name}' requires approval"
                            )
                        else:
                            run_it = True
                    # Run the tool (whichever path allowed it) while forwarding any progress
                    # it reports as tool_progress events — long media tools stream a bar
                    # instead of going silent for minutes. Non-reporting tools just resolve.
                    if run_it:
                        self._snapshot_before_edit(tc, turn_snapshots, ctx)
                        self._snapshot_before_shell(tc, shell_state, ctx)
                        async for sub in self._run_tool_with_progress(tool, tc, ctx):
                            if sub["type"] == "progress":
                                yield {
                                    "type": "tool_progress",
                                    "id": tc["id"],
                                    "name": tc["name"],
                                    "fraction": sub["fraction"],
                                    "label": sub["label"],
                                }
                            else:
                                result = sub["result"]
                    # Remember images we produced so a follow-up view_image on them is skipped.
                    if (
                        tc["name"] in ("generate_image", "edit_image")
                        and result is not None
                        and result.ok
                    ):
                        generated_images.add(result.content)
                    session.messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
                    )
                    if trace is not None:
                        step.tool_results.append(
                            {"name": tc["name"], "ok": result.ok, "content": result.content}
                        )
                    yield {
                        "type": "tool_result",
                        "id": tc["id"],
                        "name": tc["name"],
                        "ok": result.ok,
                        "content": result.content,
                    }
                    # Plan checklist (SA.3): unlike turn_diff (terminal, one-shot), a plan is
                    # mutated repeatedly within a turn, so we emit a `plan` event each time the
                    # agent calls update_plan. The authoritative list lives only here (turn-local)
                    # and in the event — NOT in session.messages — so a stale checklist isn't
                    # re-fed every iteration (history bloat / wasted context on small models).
                    if tc["name"] == "update_plan" and result.ok:
                        try:
                            yield {
                                "type": "plan",
                                "steps": normalize_steps(tc["arguments"].get("steps")),
                            }
                        except ValueError:
                            pass  # malformed args already surfaced as the tool's error result
                if trace is not None:
                    trace.steps.append(step)

            if trace is not None:
                self._trace.record(trace.finalize("max_iters"))
            yield {"type": "error", "detail": f"reached max tool iterations ({self._max_iters})"}
        except Exception as exc:
            # A turn can die mid-loop — most often a chat-template render failure when the
            # tool_calls history is fed back to the model. CancelledError / GeneratorExit are
            # BaseException (client disconnect, New chat), so they bypass this handler and are
            # NOT logged as failures. Record the error trace, then re-raise so the API layer
            # still streams its error event (outward behaviour unchanged).
            if trace is not None:
                if step is not None and not any(s is step for s in trace.steps):
                    trace.steps.append(step)
                trace.error = str(exc)
                self._trace.record(trace.finalize("error"))
            raise

    def _build_stable_system(self) -> str:
        # Stable across turns: base text + skills index only (no user-dependent content),
        # so it is a byte-identical cacheable prefix. Skills live on the tool context.
        skills = self._ctx.skills
        skills_index = skills.index_text() if skills else "(no skills available)"
        return build_system_prompt(skills_index)

    @staticmethod
    def _system_fingerprint(system: str, model: str) -> str:
        # Model is part of the key because the KV-cache is per-model: switching models
        # invalidates the cached prefix even when the prompt text is unchanged.
        return hashlib.sha256(f"{model}\x00{system}".encode()).hexdigest()

    async def _prefetch_memory(self, user_text: str) -> str:
        memory = self._ctx.memory
        return await memory.prefetch(user_text) if memory else ""

    @staticmethod
    def _messages_for_send(messages: list[dict], context_blocks: list[str]) -> list[dict]:
        """Send-time view of the conversation: reference-only blocks (current date,
        prefetched memory) ride the *latest user message*. Stored history (and the GUI) stay
        clean, and the cacheable prefix — system + all prior turns — is left byte-for-byte
        unchanged. Returns ``messages`` unchanged when there are no blocks to inject."""
        blocks = [b for b in context_blocks if b]
        if not blocks:
            return messages
        suffix = "\n\n".join(blocks)
        out = list(messages)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                m = dict(out[i])
                m["content"] = f"{m['content']}\n\n{suffix}"
                out[i] = m
                break
        return out

    @staticmethod
    def _assistant_tool_msg(text_parts: list[str], tool_calls: list[dict]) -> dict:
        return {
            "role": "assistant",
            "content": "".join(text_parts) or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        }

    def set_max_output_tokens(self, n: int) -> None:
        """Live-update the per-turn generation ceiling (GUI Settings edit; takes effect next
        turn, no restart)."""
        self._max_output_tokens = n

    def set_max_iters(self, n: int) -> None:
        """Live-update the per-turn tool-iteration budget (GUI Settings edit; the next turn's
        loop reads it, so no restart)."""
        self._max_iters = n

    def _visible_tool_schemas(self):
        """Tool schemas offered to the model, with blanket-denied tools (S5) filtered out — no
        point tempting the model to call something a rule will always refuse. Resource-specific
        denials stay visible and are enforced at call time by ``_rule_decision``."""
        schemas = self._registry.schemas()
        if self._rules:
            schemas = [
                s
                for s in schemas
                if not any(r.is_blanket_deny(s["function"]["name"]) for r in self._rules)
            ]
        return schemas or None

    def _rule_decision(self, tool, arguments: dict) -> str:
        """S5 permission decision: 'allow' (run without prompting), 'deny' (refuse), or 'ask'
        (defer to the interactive/policy approver). Safe tools and session-remembered grants
        short-circuit to allow. Among rules: deny wins outright (a broad allow can't override a
        deny — fail safe); otherwise the last matching allow/ask wins; no match -> 'ask'."""
        if not tool.needs_approval:
            return "allow"
        resource = resource_of(arguments)
        if self._ask_once and (tool.name, resource) in self._granted:
            return "allow"
        decision = "ask"  # default when nothing matches
        for rule in self._rules:
            if rule.matches(tool.name, resource):
                if rule.decision == "deny":
                    return "deny"  # deny-priority: short-circuit
                decision = rule.decision  # last-match-wins among allow/ask
        return decision

    def _remember(self, tool, arguments: dict) -> None:
        if self._ask_once:
            self._granted.add((tool.name, resource_of(arguments)))

    async def _run_tool_with_progress(self, tool, tc: dict, ctx: ToolContext):
        """Run a tool, yielding ``{"type": "progress", ...}`` for each tick it reports and a
        final ``{"type": "result", "result": ToolResult}``.

        Tools call ``ctx.on_progress(fraction, label)`` to report; for a tool that offloads
        to a worker thread (video gen) that callback fires off-loop, so we marshal ticks onto
        a queue via ``call_soon_threadsafe`` and drain them while the tool runs. The sink is
        scoped to this single run and always cleared, so non-reporting tools are unaffected.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        ctx.on_progress = lambda fraction, label="": loop.call_soon_threadsafe(
            queue.put_nowait, (fraction, label)
        )
        task = asyncio.ensure_future(self._run_tool(tool, tc, ctx))
        t0 = loop.time()
        last_emit = t0
        try:
            while not task.done():
                try:
                    fraction, label = await asyncio.wait_for(queue.get(), timeout=0.25)
                    yield {"type": "progress", "fraction": fraction, "label": label}
                    last_emit = loop.time()
                except asyncio.TimeoutError:
                    # Heartbeat: a long, silent tool (bash, web fetch) must not look frozen.
                    # Once it's run a while with no progress of its own, emit an elapsed-time
                    # tick (fraction <0 marks it indeterminate, not a real bar) so the gateway
                    # can show "working…". Suppressed while a tool reports real progress.
                    now = loop.time()
                    if now - last_emit >= self._HEARTBEAT_SECS:
                        el = int(now - t0)
                        yield {"type": "progress", "fraction": -1.0,
                               "label": f"{el // 60}:{el % 60:02d}"}
                        last_emit = now
            while not queue.empty():  # ticks that landed just before completion
                fraction, label = queue.get_nowait()
                yield {"type": "progress", "fraction": fraction, "label": label}
            yield {"type": "result", "result": task.result()}
        finally:
            ctx.on_progress = None
            task.cancel()  # no-op once done; tidies the orphan if the caller stops early

    async def _run_tool(self, tool, tc: dict, ctx: ToolContext) -> ToolResult:
        # Hook seam (P2): tool_call can veto or mutate args (post-approval, trusted), and
        # tool_result can observe or replace the outcome. Centralised here so every run
        # path (rule-allowed, interactively granted, policy-allowed) fires them uniformly.
        gate = await self._hooks.fire_tool_call(tool.name, tc["arguments"])
        if gate.block:
            return ToolResult(False, f"blocked by hook: {gate.reason}")
        try:
            result = await tool.handler(gate.arguments, ctx)
        except Exception as exc:  # never let a tool crash take down the turn
            result = ToolResult(False, f"error running {tool.name}: {exc}")
        return await self._hooks.fire_tool_result(tool.name, result)

    # Tools whose target file we snapshot to build the turn diff. shell-created files are
    # caught separately via git (see _snapshot_before_shell / _merge_shell_changes).
    _EDIT_TOOLS = ("write_file", "edit_file")
    # How long a tool may run silently before _run_tool_with_progress emits an elapsed-time
    # heartbeat. Tests override this on the instance to fire it quickly.
    _HEARTBEAT_SECS = 5.0

    def _snapshot_before_edit(
        self, tc: dict, snapshots: dict[str, bytes | None], ctx: ToolContext
    ) -> None:
        """Record a write/edit target's bytes before its first touch this turn (None when the
        file is new), so the turn diff can show before→after. Only the earliest snap wins."""
        if tc["name"] not in self._EDIT_TOOLS:
            return
        raw = (tc.get("arguments") or {}).get("path")
        if not raw:
            return
        path = Path(raw)
        if not path.is_absolute():
            path = ctx.cwd / path
        key = str(path)
        if key in snapshots:
            return
        try:
            snapshots[key] = path.read_bytes()
        except OSError:
            snapshots[key] = None  # new file — didn't exist before this turn

    def _snapshot_before_shell(
        self, tc: dict, state: dict, ctx: ToolContext
    ) -> None:
        """On the first bash call, record git's pre-shell baseline so the turn diff can later
        show what shell changed. Snapshots only the already-dirty files' bytes (their true
        'before'); clean files fall back to their committed content at merge time."""
        if tc["name"] != "bash" or state.get("captured"):
            return
        state["captured"] = True
        root = repo_root(ctx.cwd)
        state["root"] = root
        if root is None:
            return  # not a git repo — shell changes simply aren't captured
        pre = dirty_paths(root)
        pre_bytes: dict[str, bytes | None] = {}
        for rel in pre:
            try:
                pre_bytes[rel] = (root / rel).read_bytes()
            except OSError:
                pre_bytes[rel] = None
        state["pre_bytes"] = pre_bytes

    def _merge_shell_changes(self, state: dict, snapshots: dict[str, bytes | None]) -> None:
        """Fold shell-touched files into the snapshots so they flow through the diff builder.
        'before' = pre-shell bytes for files already dirty, else committed (HEAD) bytes for
        files clean before shell, else None for new untracked files. Files unchanged net of
        the turn are dropped later by the diff builder (before == after)."""
        root = state.get("root")
        if root is None:
            return
        pre_bytes = state.get("pre_bytes", {})
        for rel in dirty_paths(root):
            abspath = str(root / rel)
            if abspath in snapshots:
                continue  # write/edit already captured this file
            snapshots[abspath] = (
                pre_bytes[rel] if rel in pre_bytes else head_bytes(root, rel)
            )

    def _turn_diff_event(
        self, snapshots: dict[str, bytes | None], ctx: ToolContext
    ) -> dict | None:
        """Read each snapshotted file's current bytes and build a turn_diff event, or None if
        nothing net-changed (e.g. an edit that wrote identical content, or a binary-only touch
        that produced no diff lines)."""
        if not snapshots:
            return None
        pairs: dict[str, tuple[bytes | None, bytes | None]] = {}
        for key, before in snapshots.items():
            path = Path(key)
            try:
                after = path.read_bytes()
            except OSError:
                after = None
            try:
                display = str(path.relative_to(ctx.cwd))
            except ValueError:
                display = str(path)
            pairs[display] = (before, after)
        changes = build_turn_changes(pairs)
        if not changes.files:
            return None
        return {
            "type": "turn_diff",
            "summary": changes.summary(),
            "files": [
                {
                    "path": f.path,
                    "status": f.status,
                    "additions": f.additions,
                    "deletions": f.deletions,
                }
                for f in changes.files
            ],
            "diff": changes.diff,
        }
