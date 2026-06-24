from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import AsyncIterator

from assistant.tools.approval import ApprovalPolicy, Rule, resource_of
from assistant.tools.base import ToolContext, ToolResult
from assistant.tools.registry import ToolRegistry

from .compaction import CompactionManager
from .hooks import HookRegistry
from .llm_client import AsyncLLM
from .prompt import build_system_prompt, wrap_memory_context
from .session import Session
from .tokens import estimate_messages_tokens, estimate_tokens

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
    ):
        self._llm = llm
        self._registry = registry
        self._approver = approver
        self._ctx = ctx
        self._max_iters = max_iters
        self._compaction = compaction
        self._max_output_tokens = max_output_tokens
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
    ) -> AsyncIterator[dict]:
        # Per-run approver lets the Telegram gateway inject an interactive (inline
        # button) approver while the desktop path uses the default policy approver.
        effective_approver = approver or self._approver
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

        for _ in range(self._max_iters):
            text_parts: list[str] = []
            tool_calls: list[dict] | None = None

            send_messages = self._messages_for_send(session.messages, memory_block)
            async for ev in self._llm.stream_chat(
                send_messages, model, tools=tools, max_tokens=self._max_output_tokens
            ):
                if ev["type"] == "text":
                    text_parts.append(ev["content"])
                    yield {"type": "assistant_delta", "content": ev["content"]}
                elif ev["type"] == "tool_calls":
                    tool_calls = ev["tool_calls"]

            if not tool_calls:
                answer = "".join(text_parts)
                session.add_assistant(answer)
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
            for tc in tool_calls:
                yield {
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                }
                tool = self._registry.get(tc["name"])
                if tool is None:
                    result = ToolResult(False, f"unknown tool: {tc['name']}")
                else:
                    # S5: rules + ask-once decide allow/deny before any prompt; only an
                    # unresolved "ask" falls through to the interactive/policy approver.
                    decision = self._rule_decision(tool, tc["arguments"])
                    if decision == "allow":
                        result = await self._run_tool(tool, tc)
                    elif decision == "deny":
                        result = ToolResult(False, f"denied by rule: tool '{tool.name}'")
                    elif tool.needs_approval and getattr(
                        effective_approver, "interactive", False
                    ):
                        # Interactive approver (GUI/HTTP): surface the request and wait for
                        # an out-of-band decision (POST /chat/approve) before running.
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
                            result = await self._run_tool(tool, tc)
                        else:
                            result = ToolResult(
                                False, f"denied: tool '{tool.name}' requires approval"
                            )
                    elif not await effective_approver.approve(tool, tc["arguments"]):
                        result = ToolResult(
                            False, f"denied: tool '{tool.name}' requires approval"
                        )
                    else:
                        result = await self._run_tool(tool, tc)
                session.messages.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": result.content}
                )
                yield {
                    "type": "tool_result",
                    "id": tc["id"],
                    "name": tc["name"],
                    "ok": result.ok,
                    "content": result.content,
                }

        yield {"type": "error", "detail": f"reached max tool iterations ({self._max_iters})"}

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
    def _messages_for_send(messages: list[dict], memory_block: str) -> list[dict]:
        """Send-time view of the conversation: prefetched memory rides the *latest user
        message* as a reference-only block. Stored history (and the GUI) stay clean, and
        the cacheable prefix — system + all prior turns — is left byte-for-byte unchanged.
        Returns ``messages`` unchanged when there is no memory to inject."""
        if not memory_block:
            return messages
        out = list(messages)
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                m = dict(out[i])
                m["content"] = f"{m['content']}\n\n{wrap_memory_context(memory_block)}"
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

    async def _run_tool(self, tool, tc: dict) -> ToolResult:
        # Hook seam (P2): tool_call can veto or mutate args (post-approval, trusted), and
        # tool_result can observe or replace the outcome. Centralised here so every run
        # path (rule-allowed, interactively granted, policy-allowed) fires them uniformly.
        gate = await self._hooks.fire_tool_call(tool.name, tc["arguments"])
        if gate.block:
            return ToolResult(False, f"blocked by hook: {gate.reason}")
        try:
            result = await tool.handler(gate.arguments, self._ctx)
        except Exception as exc:  # never let a tool crash take down the turn
            result = ToolResult(False, f"error running {tool.name}: {exc}")
        return await self._hooks.fire_tool_result(tool.name, result)
