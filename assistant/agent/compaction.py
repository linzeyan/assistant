"""Conversation compaction (spring1 S6).

Long sessions — especially Telegram chats that run for days — grow until they exceed the
local model's context window. When the estimated footprint crosses ``window - reserve``,
the oldest turns are summarized by one LLM call into a compact brief, the most-recent turns
are kept verbatim, and the live ``messages`` list is rebuilt as ``[system, summary, recent]``.

Two correctness invariants:
- **Split-turn safe**: the boundary between summarized and kept messages is aligned back to a
  turn start (a user message), so an assistant ``tool_calls`` message and its ``tool`` results
  are never separated, and no half-turn is summarized.
- **No data loss**: the original messages a compaction replaced are archived on
  ``session.compactions`` (recoverable from disk); an empty/failed summary aborts the
  compaction rather than dropping history.

The model's context window is auto-detected (``llm.context_window``) with a config fallback.
"""

from __future__ import annotations

import time

from .llm_client import AsyncLLM
from .session import Session
from .tokens import estimate_messages_tokens

_SUMMARY_SYS = (
    "You compress earlier parts of a conversation into a faithful, terse brief that lets the "
    "assistant continue seamlessly. Preserve concrete facts, file paths, decisions, and open "
    "tasks. Do not invent anything. Output only the requested Markdown sections."
)

_SUMMARY_USER = """Summarize the earlier conversation below into EXACTLY these Markdown sections, terse and factual:

## Goal
## Constraints
## Progress
## Key decisions
## Next steps

Earlier conversation:
{transcript}"""

_SUMMARY_BANNER = (
    "<conversation-summary reference-only>\n"
    "Earlier turns were compacted to fit the context window. Summary of what came before:\n\n"
    "{summary}\n"
    "</conversation-summary>"
)


class CompactionManager:
    """Threshold-triggered, split-turn-safe conversation summarizer."""

    def __init__(
        self,
        llm: AsyncLLM,
        *,
        context_window_fallback: int,
        reserve_tokens: int,
        keep_recent_tokens: int,
    ):
        self._llm = llm
        self._fallback = context_window_fallback
        self._reserve = reserve_tokens
        self._keep_recent = keep_recent_tokens

    async def maybe_compact(self, session: Session, model: str | None) -> dict | None:
        """Compact only if the conversation exceeds ``window - reserve``. Returns a
        ``compaction`` event dict when it compacted, else None."""
        if not model:
            return None
        window = await self._window(model)
        if estimate_messages_tokens(session.messages) <= window - self._reserve:
            return None
        return await self._compact(session, model, window)

    async def force_compact(self, session: Session, model: str | None) -> dict | None:
        """Compact regardless of the threshold (manual ``/compact``). Returns the event dict,
        or None when there is nothing old enough to safely summarize."""
        if not model:
            return None
        return await self._compact(session, model, await self._window(model))

    async def _window(self, model: str) -> int:
        return await self._llm.context_window(model) or self._fallback

    async def _compact(self, session: Session, model: str, window: int) -> dict | None:
        plan = self._plan(session.messages)
        if plan is None:
            return None
        system, older, recent = plan
        summary = await self._summarize(older, model)
        if not summary:
            return None  # model returned nothing — never drop history on an empty summary
        tokens_before = estimate_messages_tokens(session.messages)
        summary_msg = {"role": "user", "content": _SUMMARY_BANNER.format(summary=summary)}
        session.messages = system + [summary_msg] + recent
        tokens_after = estimate_messages_tokens(session.messages)
        session.compactions.append(
            {
                "summary": summary,
                "dropped": older,  # archived verbatim — nothing is lost
                "dropped_count": len(older),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "at": time.time(),
            }
        )
        return {
            "type": "compaction",
            "dropped": len(older),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "context_window": window,
        }

    def _plan(self, messages: list[dict]) -> tuple[list, list, list] | None:
        """Split into (system, older-to-summarize, recent-to-keep). Returns None when no
        clean, worthwhile split exists."""
        system = messages[:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(system):]
        if not body:
            return None
        # Walk from the end accumulating tokens until ~keep_recent is reserved verbatim.
        acc, cut = 0, None
        for i in range(len(body) - 1, -1, -1):
            acc += estimate_messages_tokens([body[i]])
            if acc >= self._keep_recent:
                cut = i
                break
        if cut is None:
            return None  # whole body fits in keep-recent — nothing old enough to drop
        # Split-turn safety: move the boundary BACK to the nearest turn start (user message),
        # so `recent` keeps whole turns and no tool_calls/tool pair is split by the summary.
        while cut > 0 and body[cut].get("role") != "user":
            cut -= 1
        older, recent = body[:cut], body[cut:]
        if not older or not recent:
            return None
        return system, older, recent

    async def _summarize(self, older: list[dict], model: str) -> str:
        messages = [
            {"role": "system", "content": _SUMMARY_SYS},
            {"role": "user", "content": _SUMMARY_USER.format(transcript=self._render(older))},
        ]
        parts: list[str] = []
        async for ev in self._llm.stream_chat(messages, model, tools=None):
            if ev.get("type") == "text":
                parts.append(ev["content"])
        return "".join(parts).strip()

    @staticmethod
    def _render(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            content = m.get("content") or ""
            tcs = m.get("tool_calls")
            if tcs:
                names = ", ".join(tc.get("function", {}).get("name", "?") for tc in tcs)
                content = f"{content} [tool calls: {names}]".strip()
            lines.append(f"{m.get('role', '?')}: {content}")
        return "\n".join(lines)
