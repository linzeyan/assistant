"""Per-turn trace capture (spring2 P0 — make failure visible).

The product's core pain is that local-model turns "don't always succeed" (chat / web
search / image / code), but there's no way to see WHERE a turn died. This records each
``AgentLoop.run`` turn — the model's (post-suppression) text, the tool calls parsed from
it, and each tool's result — so a human can scan a session and tell apart failure modes
that need completely different fixes:

  * model never emitted a tool call          — model decision / quality (human judges:
    "should it have called one here?" by reading ``model_text``)
  * model emitted one but the parser missed  — ``parse_miss``: the markup leaks into
    ``model_text`` because the stream flushes unparsed markup back as text
    (``mlx_service``: no parsed call → remainder emitted as text)
  * the tool ran and errored                 — ``tool_error``
  * the tool succeeded but the model ignored — read ``model_text`` by eye

P0 deliberately RECORDS only — it fixes nothing and judges nothing about answer quality.
That is the whole measure-before-fix discipline (spring2 §3, §8 invariant #7). Persistence
mirrors ``SessionStore``: file-per-turn JSON, atomic writes, degrade to memory-only without
a dir, capped to bound disk growth (no separate GC at P0).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from assistant.models.tool_parsing import TOOL_MARKERS

# Bump when the on-disk JSON shape changes incompatibly.
SCHEMA_VERSION = 1

# A tool call the parser failed to recognise leaks into the visible text as raw markup.
# These are the openers worth flagging; ``<function=`` is the nested-XML form (the N3 class).
_ATTEMPT_MARKERS = (*TOOL_MARKERS, "<function=")


def _now() -> float:
    return time.time()


def looks_like_tool_attempt(text: str) -> bool:
    """True when ``text`` carries an UNPARSED tool-call attempt — the signal for a ``parse_miss``
    (the model tried to call a tool but it leaked back as text instead of being parsed).

    A marker counts only when something follows it: a bare trailing ``<tool_call>`` with nothing
    after it — a model that finished a complete answer and emitted a stray opener — is NOT an
    attempt, and flagging it would mislabel a perfectly answered turn as parse_miss (a real
    false-positive the A1 harness surfaced on a Qwen3-Coder answer ending in a dangling
    ``<tool_call>``)."""
    return any(
        (idx := text.find(marker)) != -1 and text[idx + len(marker):].strip()
        for marker in _ATTEMPT_MARKERS
    )


def _summary_of(data: dict) -> dict:
    """Scannable row for the turn list: enough to spot a failure without the full bodies."""
    return {
        "turn_id": data.get("turn_id"),
        "session_id": data.get("session_id"),
        "model": data.get("model"),
        "outcome": data.get("outcome"),
        "steps": len(data.get("steps") or []),
        "user_text": (data.get("user_text") or "")[:80],
        "created_at": data.get("created_at") or 0.0,
    }


@dataclass
class TraceStep:
    """One LLM iteration in a turn. ``model_text`` is the user-visible (post-suppression)
    text; a parser miss shows up here as leaked markup (see module docstring)."""

    model_text: str = ""
    parsed_calls: list[dict] = field(default_factory=list)  # [{"name","arguments"}]
    tool_results: list[dict] = field(default_factory=list)  # [{"name","ok","content"}]

    def to_dict(self) -> dict:
        return {
            "model_text": self.model_text,
            "parsed_calls": self.parsed_calls,
            "tool_results": self.tool_results,
        }


@dataclass
class TurnTrace:
    """One ``run()`` invocation: a user message and the model/tool steps it drove, with a
    coarse ``outcome`` so a human can scan a session for failures fast."""

    turn_id: str
    session_id: str
    model: str
    user_text: str
    steps: list[TraceStep] = field(default_factory=list)
    final_text: str = ""
    outcome: str = "answered"
    error: str | None = None  # exception text when the turn died mid-loop (outcome=="error")
    schema_version: int = SCHEMA_VERSION
    created_at: float = field(default_factory=_now)

    @staticmethod
    def new(session_id: str, model: str, user_text: str) -> "TurnTrace":
        return TurnTrace(uuid.uuid4().hex, session_id, model, user_text)

    def finalize(self, terminal: str) -> "TurnTrace":
        """Classify ``outcome`` from coarse, high-confidence signals so a human can scan for
        failures. ``terminal`` is the loop's exit reason ('answered' | 'max_iters' | 'error').
        A non-normal exit wins (``error`` = an exception killed the turn mid-loop, e.g. a
        chat-template render failure — the single most common real failure); else a
        leaked-but-unparsed call (``parse_miss``) is flagged ahead of a tool error, since a
        missed call means the tool never ran. 'answered' does NOT mean the answer was good —
        that's a human call from ``model_text``."""
        if terminal != "answered":
            self.outcome = terminal
        elif any(
            not s.parsed_calls and looks_like_tool_attempt(s.model_text) for s in self.steps
        ):
            self.outcome = "parse_miss"
        elif any(not r["ok"] for s in self.steps for r in s.tool_results):
            self.outcome = "tool_error"
        else:
            self.outcome = "answered"
        return self

    def summary(self) -> dict:
        return _summary_of(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "model": self.model,
            "user_text": self.user_text,
            "steps": [s.to_dict() for s in self.steps],
            "final_text": self.final_text,
            "outcome": self.outcome,
            "error": self.error,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
        }


class TraceStore:
    """Durable per-turn trace registry: in-memory cache + file-per-turn JSON, mirroring
    ``SessionStore``. Without a dir it's memory-only (tests, or ``trace_enabled=false``).
    Best-effort — a trace write must never break a live turn. Capped to the most recent
    ``max_turns`` to bound disk/memory growth (P0 has no separate GC; this ring is enough)."""

    def __init__(self, trace_dir: Path | str | None = None, max_turns: int = 1000) -> None:
        self._cache: dict[str, TurnTrace] = {}
        self._max_turns = max_turns
        self._dir = Path(trace_dir) if trace_dir else None
        if self._dir is not None:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._dir = None  # degrade to memory-only rather than crash startup

    def record(self, trace: TurnTrace) -> None:
        self._cache[trace.turn_id] = trace
        self._trim_cache()
        if self._dir is None:
            return
        path = self._dir / f"{trace.turn_id}.json"
        tmp = path.with_name(f"{trace.turn_id}.json.tmp")
        try:
            tmp.write_text(
                json.dumps(trace.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        except OSError:
            # Tracing is best-effort: a write failure must not break the live turn.
            tmp.unlink(missing_ok=True)
            return
        self._prune_disk()

    def get(self, turn_id: str) -> dict | None:
        if turn_id in self._cache:
            return self._cache[turn_id].to_dict()
        if self._dir is None:
            return None
        return self._read_json(self._dir / f"{turn_id}.json")

    def clear(self) -> int:
        """Wipe every recorded trace, memory and disk (Settings ▸ "Clear traces"). Returns how
        many turns were dropped. Best-effort per file — a locked/vanished file must not fail the
        wipe; stray ``.json.tmp`` from a dead write is swept too."""
        dropped = set(self._cache)
        self._cache.clear()
        if self._dir is not None:
            for path in self._dir.glob("*.json"):
                with contextlib.suppress(OSError):
                    path.unlink()
                    dropped.add(path.stem)
            for tmp in self._dir.glob("*.json.tmp"):
                tmp.unlink(missing_ok=True)
        return len(dropped)

    def list_for_session(self, session_id: str) -> list[dict]:
        """Turn summaries for one session, newest first. Merges the in-memory cache with
        on-disk traces not currently cached."""
        summaries: dict[str, dict] = {
            t.turn_id: t.summary()
            for t in self._cache.values()
            if t.session_id == session_id
        }
        if self._dir is not None:
            for path in self._dir.glob("*.json"):
                tid = path.stem
                if tid in summaries:
                    continue
                data = self._read_json(path)
                if data is not None and data.get("session_id") == session_id:
                    summaries[tid] = _summary_of(data)
        return sorted(summaries.values(), key=lambda d: d["created_at"], reverse=True)

    # --- internals ---

    def _trim_cache(self) -> None:
        excess = len(self._cache) - self._max_turns
        if excess <= 0:
            return
        for tid in sorted(self._cache, key=lambda k: self._cache[k].created_at)[:excess]:
            self._cache.pop(tid, None)

    def _prune_disk(self) -> None:
        if self._dir is None:
            return
        files = list(self._dir.glob("*.json"))
        excess = len(files) - self._max_turns
        if excess <= 0:
            return
        files.sort(key=lambda p: p.stat().st_mtime)  # oldest first
        for p in files[:excess]:
            p.unlink(missing_ok=True)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # corrupt or missing → treat as absent, never crash
