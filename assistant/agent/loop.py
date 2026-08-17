from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from assistant.models.tool_parsing import strip_think_blocks
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
    wrap_plan_context,
    wrap_referenced_paths,
    wrap_workspace_context,
)
from .session import Session
from .tokens import cut_at_tokens, estimate_messages_tokens, estimate_tokens
from .trace import TraceStep, TraceStore, TurnTrace

log = logging.getLogger("assistant")

# Thrash guard (B2): abort a turn after this many consecutive *identical* tool calls that made no
# progress. A1 telemetry shows healthy turns chain several DIFFERENT calls (7+ on a hard web task),
# so the signal isn't call count — it's the same (name, args) repeating. Three ignored repeats is a
# stuck loop, not work. A backstop constant, not a user knob: max_iters is the tunable budget.
_THRASH_REPEAT_CAP = 3

# Context fuse (N94): combined estimated-token cap on the tool results one iteration's batch may
# inject into the conversation. Compaction (S6) only trims BETWEEN turns, so an in-turn bomb —
# several fetch_url calls, or one dense CJK page — had no bound at all: one such batch grew a
# turn by ~20k tokens on a 65.7GB model and ground the whole machine into swap. 6000 ≈ four
# max-size fetch_url results: roomy for real multi-tool steps, small enough that no single step
# can blow the context. A backstop, not a knob.
_TOOL_RESULT_BATCH_TOKENS = 6000

# Degenerate-generation guard: a model that falls into a repetition loop emits the same
# sentences until it hits max_output_tokens, and because the turn ends with text and no tool
# call, the loop above treats that wall of repeats as the answer and reports success. Seen on a
# 35B: twenty minutes and the whole output budget spent on "OK, let me just write the code now."
# repeated eighty times, with not one file written.
#
# Two conditions, because either alone has a false positive. A verbatim 240-character tail
# recurring several times is the shape of the loop — but source code repeats blocks too, so
# recurrence alone would truncate a legitimate file. The second condition is that those repeats
# account for most of the reply: a repeated block inside otherwise-new text is someone writing
# code, whereas a reply that IS its own tail six times over has stopped saying anything.
#
# Checked only past _DEGENERATE_MIN_CHARS, and only every _DEGENERATE_EVERY deltas, so short
# answers never pay for it and long ones do not pay per token.
_DEGENERATE_TAIL = 240
_DEGENERATE_REPEATS = 6
_DEGENERATE_COVERAGE = 0.5
_DEGENERATE_MIN_CHARS = 4000
_DEGENERATE_EVERY = 50


def _is_degenerate(text: str) -> bool:
    """Whether `text` has collapsed into repeating its own tail."""
    if len(text) < _DEGENERATE_MIN_CHARS:
        return False
    tail = text[-_DEGENERATE_TAIL:]
    repeats = text.count(tail)
    if repeats < _DEGENERATE_REPEATS:
        return False
    return repeats * _DEGENERATE_TAIL >= _DEGENERATE_COVERAGE * len(text)

# Path-like substrings in a user turn: something containing a slash (a/b, /abs, ~/x, src/),
# a bare filename.ext (config.toml, README.md), or a well-known extensionless name (Makefile,
# Dockerfile). re.ASCII keeps \w to [A-Za-z0-9_] so CJK is NOT a word char — a path pressed
# against Chinese ("看下/Users/…") still extracts as "/Users/…". The slash branches require a real
# path char on one side so a lone "/" in prose (e.g. "read / write") never matches. This only
# *finds candidates*; existence on disk is the real gate in _referenced_paths, so a token that
# merely looks path-like but isn't there is dropped and false positives never reach the model.
_KNOWN_BARE = "Makefile|Dockerfile|README|LICENSE|CHANGELOG|Rakefile|Gemfile|Justfile|Procfile"
_PATH_LIKE = re.compile(
    rf"[\w.~-]+/[\w./~-]*|/[\w./~-]+|(?<![\w./-])[\w-]+\.[A-Za-z][\w]*"
    rf"|(?<![\w./-])(?:{_KNOWN_BARE})(?![\w.])",
    re.ASCII,
)


def _referenced_paths(text: str, cwd: Path, *, limit: int = 5) -> list[str]:
    """Local paths the user named this turn that actually exist on disk. A weak-at-tools model,
    told to "看下 <path>", tends to answer from imagination instead of opening the file (observed:
    it fabricated a whole Makefile without reading the real one). Surfacing the concrete existing
    paths lets the loop nudge it to read them first — see wrap_referenced_paths. Existence is the
    gate, so extraction can be liberal without risking bogus nudges."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in _PATH_LIKE.findall(text or ""):
        tok = tok.strip("'\"`").rstrip(".,;:")  # trim quotes + trailing sentence punctuation
        if not tok:
            continue
        if tok.startswith("~"):
            p = Path(tok).expanduser()
        elif tok.startswith("/"):
            p = Path(tok)
        else:
            p = cwd / tok
        try:
            if not p.exists():
                continue
        except OSError:  # overlong / malformed path — treat as not present
            continue
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= limit:  # cap: name a few, don't flood the turn
            break
    return out


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
        turn_timeout_s: float | None = None,
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
        # Per-turn wall-clock budget (B1); None = unlimited. See the deadline check in run().
        self._turn_timeout_s = turn_timeout_s
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
        max_iters: int | None = None,
    ) -> AsyncIterator[dict]:
        # Per-run approver lets the Telegram gateway inject an interactive (inline
        # button) approver while the desktop path uses the default policy approver.
        effective_approver = approver or self._approver
        # Per-request tool-iteration budget (H7): a single hard task can raise the ceiling for one
        # turn without changing the global default. Falls back to the configured default; a non-
        # positive value is ignored so callers can't disable the backstop entirely.
        iters = max_iters if (max_iters and max_iters > 0) else self._max_iters
        # Per-run context copy (never a mutation of the shared ctx, so concurrent turns stay
        # isolated): the turn's model rides it — a tool that spawns model work (spawn_subagents)
        # must target the conversation's own model — and Telegram's per-chat cwd overrides the
        # desktop/HTTP default.
        ctx = replace(self._ctx, model=model)
        if cwd is not None:
            ctx = replace(ctx, cwd=Path(cwd))
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
        tools = self._visible_tool_schemas(ctx)
        # Reference-only blocks injected into the latest user message at send time (never the
        # cacheable system prefix, S3). The date is stamped once per turn so a local model —
        # which has no clock — stops hallucinating "today" from its training cutoff.
        referenced = _referenced_paths(user_text, ctx.cwd)
        # A finished plan stops riding; the model replaces the carried plan wholesale on its
        # next update_plan call. Snapshotted at turn start — mid-turn updates reach the model
        # through the tool acks, not this block.
        carried_plan = (
            session.plan
            if session.plan and any(s.get("status") != "completed" for s in session.plan)
            else None
        )
        context_blocks = [
            wrap_datetime_context(datetime.now().astimezone()),
            wrap_workspace_context(ctx.cwd),
            wrap_plan_context(carried_plan) if carried_plan else "",
            wrap_memory_context(memory_block) if memory_block else "",
            # An explicit "看下 <path>" is the strongest read-me signal; a weak model still guesses
            # the file's contents without this in-context nudge (see _referenced_paths).
            wrap_referenced_paths(referenced) if referenced else "",
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
        # Thrash guard (B2). A repeated *idempotent* call (read-only = no approval) makes no
        # progress, so we replay its earlier result instead of re-running it, and count consecutive
        # no-progress calls — too many in a row aborts the turn (the model stuck in a loop, ignoring
        # the replayed result) rather than burning every iteration. A *mutating* call (needs
        # approval) is never silently deduped (its side effect may be intended), and any mutation
        # invalidates the replay cache + the seen set, since a re-read after a write must run for
        # real (the file changed). idempotent_cache holds successful read results for replay; the
        # seen set drives the no-progress counter for both tool kinds.
        seen_calls: set[tuple[str, str]] = set()
        idempotent_cache: dict[tuple[str, str], str] = {}
        no_progress = 0
        # Turn-level wall-clock budget (B1): None = unlimited. Checked BETWEEN iterations —
        # bounding a runaway tool-call loop, the realistic "stuck turn" — rather than mid-
        # generation: MLX has no token-level interrupt (mlx_service awaits the worker thread in
        # its finally), and a single generation is already bounded by max_output_tokens. So we
        # stop before starting the next model call, not inside one. Loud error, never a silent stop.
        _clock = asyncio.get_running_loop().time
        deadline = _clock() + self._turn_timeout_s if self._turn_timeout_s else None

        try:
            for _ in range(iters):
                if deadline is not None and _clock() > deadline:
                    if trace is not None:
                        self._trace.record(trace.finalize("timeout"))
                    yield {
                        "type": "error",
                        "detail": f"turn exceeded its {self._turn_timeout_s:g}s time limit",
                    }
                    return
                text_parts: list[str] = []
                tool_calls: list[dict] | None = None
                step = TraceStep() if trace is not None else None

                send_messages = self._messages_for_send(session.messages, context_blocks)
                degenerate = False
                deltas = 0
                async for ev in self._llm.stream_chat(
                    send_messages, model, tools=tools, max_tokens=self._max_output_tokens
                ):
                    if ev["type"] == "text":
                        text_parts.append(ev["content"])
                        yield {"type": "assistant_delta", "content": ev["content"]}
                        # Joining the whole reply per token would make a long answer
                        # quadratic, and a repetition loop takes thousands of tokens to
                        # become one — there is nothing to catch early.
                        deltas += 1
                        if deltas % _DEGENERATE_EVERY == 0 and _is_degenerate("".join(text_parts)):
                            degenerate = True
                            break
                    elif ev["type"] == "tool_calls":
                        tool_calls = ev["tool_calls"]
                    elif ev["type"] == "usage":
                        # Token counts for the Anthropic compat route only; the GUI/gateway SSE has
                        # no use for them. Swallow here so it doesn't reach the client. (A future
                        # token-accounting feature would consume ev["input_tokens"]/["output_tokens"].)
                        continue
                    else:
                        # Forward any other event the model layer emits (e.g. Fusion's panel
                        # tool_progress) straight to the gateway/SSE consumer.
                        yield ev

                if degenerate:
                    if trace is not None:
                        self._trace.record(trace.finalize("error"))
                    yield {
                        "type": "error",
                        "detail": "the model started repeating itself and was stopped",
                    }
                    return

                if not tool_calls:
                    answer = "".join(text_parts)
                    # A turn that produced only reasoning decided nothing: it called no
                    # tool and said nothing outside <think>. Handing that back as the
                    # answer gives the caller the model's scratchpad and reports a
                    # failure as a success — which is exactly what a tool call the
                    # parser could not read looks like from here. Measured on
                    # gpt-oss-120b: five calls parsed, then it rehearsed the sixth
                    # inside its reasoning without ever emitting the recipient header,
                    # the loop saw no calls, and the turn "succeeded" having done a
                    # third of the work.
                    if not strip_think_blocks(answer).strip():
                        if trace is not None:
                            self._trace.record(trace.finalize("error"))
                        yield {
                            "type": "error",
                            "detail": "the model ended its turn without answering or "
                            "calling a tool — its reply was reasoning only, which "
                            "usually means a tool call it meant to make could not be "
                            "parsed",
                        }
                        return
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
                batch_tokens = 0  # N94: this iteration's injected tool-result total
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
                    # B2: identity of this call for thrash detection — name + canonicalised args.
                    _key = (
                        tc["name"],
                        json.dumps(tc["arguments"], sort_keys=True, ensure_ascii=False, default=str),
                    )
                    no_progress = no_progress + 1 if _key in seen_calls else 0
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
                    elif tool is not None and not tool.needs_approval and _key in idempotent_cache:
                        # B2: an identical read-only call already succeeded this turn (and nothing
                        # has mutated state since — a mutation clears this cache). Replay the earlier
                        # result instead of re-running, and tell the model it's repeating so it
                        # changes course or finishes.
                        result = ToolResult(
                            True,
                            idempotent_cache[_key]
                            + "\n\n(repeated call — returning the earlier result; do something "
                            "different or finish.)",
                        )
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
                    # Approval audit (H3): one structured line per security-relevant (approval-
                    # gated) decision, so allow/deny outcomes on mutating tools are reviewable after
                    # the fact regardless of which path (rule / ask-once / interactive / policy)
                    # decided. Read-only auto-allows and cache replays aren't security-interesting.
                    if tool is not None and tool.needs_approval:
                        log.info(
                            "approval audit: tool=%s resource=%r decision=%s",
                            tool.name,
                            resource_of(tc["arguments"]),
                            "allow" if run_it else "deny",
                        )
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
                    # B2 bookkeeping: remember this call so an identical repeat is detected. A
                    # successful read is cached for replay; a successful mutation instead invalidates
                    # the caches (a later re-read must run for real — the state changed) and clears
                    # the no-progress count, since the mutation WAS progress.
                    seen_calls.add(_key)
                    if tool is not None and result is not None and result.ok:
                        if tool.needs_approval:
                            idempotent_cache.clear()
                            seen_calls = {_key}
                            no_progress = 0
                        else:
                            idempotent_cache.setdefault(_key, result.content)
                    # Remember images we produced so a follow-up view_image on them is skipped.
                    if (
                        tc["name"] in ("generate_image", "edit_image")
                        and result is not None
                        and result.ok
                    ):
                        generated_images.add(result.content)
                    # Context fuse (N94): cap what this batch feeds back to the model. Past the
                    # budget, results are cut — never dropped, since the API requires a reply
                    # per tool_call_id. The full content still reaches the trace and the
                    # tool_result event below; only what the MODEL sees is bounded.
                    content = result.content
                    remaining = _TOOL_RESULT_BATCH_TOKENS - batch_tokens
                    if remaining <= 0:
                        content = (
                            "[not shown: this step's combined tool results exceeded the "
                            "context budget — narrow the request or fetch less at once]"
                        )
                    elif estimate_tokens(content) > remaining:
                        content = cut_at_tokens(content, remaining) + (
                            f"\n...[truncated: combined tool results this step hit the "
                            f"{_TOOL_RESULT_BATCH_TOKENS}-token budget]"
                        )
                    batch_tokens += estimate_tokens(content)
                    session.messages.append(
                        {"role": "tool", "tool_call_id": tc["id"], "content": content}
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
                            plan_steps = normalize_steps(tc["arguments"].get("steps"))
                        except ValueError:
                            pass  # malformed args already surfaced as the tool's error result
                        else:
                            # Durable copy (still outside messages): an unfinished plan rides
                            # the NEXT turn as a reference block, so the model keeps its own
                            # checklist across turns instead of restarting the task.
                            session.plan = plan_steps
                            yield {"type": "plan", "steps": plan_steps}
                if trace is not None:
                    trace.steps.append(step)

                # B2: the model has repeated an identical call this many times in a row despite
                # being told it's repeating — it's stuck. Stop loudly instead of burning the rest of
                # the iteration budget on the same call.
                if no_progress >= _THRASH_REPEAT_CAP:
                    if trace is not None:
                        self._trace.record(trace.finalize("thrash"))
                    yield {
                        "type": "error",
                        "detail": f"stopped: the model repeated an identical tool call "
                        f"{no_progress} times in a row without making progress",
                    }
                    return

            if trace is not None:
                self._trace.record(trace.finalize("max_iters"))
            yield {"type": "error", "detail": f"reached max tool iterations ({iters})"}
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

    def set_turn_timeout(self, seconds: float | None) -> None:
        """Live-update the per-turn wall-clock budget (GUI Settings edit; the next turn reads it,
        no restart). None / 0 disables it."""
        self._turn_timeout_s = seconds or None

    def _visible_tool_schemas(self, ctx: ToolContext):
        """Tool schemas offered to the model, with two schema-time filters:

        - S5: blanket-denied tools are dropped — no point tempting the model to call something a
          rule will always refuse (resource-specific denials stay visible, enforced at call time).
        - G/S13: a tool with a ``check_fn`` is dropped when it reports unavailable for this turn's
          context, so an unloaded/uninstalled vision/audio/video backend never reaches the model —
          it can't be called and can't fail at runtime (death mode 3)."""
        schemas = []
        for tool in self._registry.all():
            if tool.check_fn is not None and not tool.check_fn(ctx):
                continue
            if self._rules and any(r.is_blanket_deny(tool.name) for r in self._rules):
                continue
            schemas.append(tool.to_openai())
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
