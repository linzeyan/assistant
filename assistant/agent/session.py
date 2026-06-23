from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

# Bump when the on-disk JSON shape changes incompatibly; load tolerates older files by
# filling defaults rather than failing.
SCHEMA_VERSION = 1


def _now() -> float:
    return time.time()


@dataclass
class Session:
    """A single conversation: the OpenAI-style message list sent to the model each turn,
    plus the S1 keystone metadata that makes a session durable and resumable.

    The fields beyond ``messages`` exist so persistence can grow into prefix stability
    (``system_prompt`` / ``system_fingerprint`` reuse) and compaction lineage
    (``parent_session_id``) without another schema migration — see spring1.md S1–S3/S6.
    """

    id: str
    model: str | None = None
    messages: list[dict] = field(default_factory=list)
    title: str | None = None
    system_prompt: str | None = None
    system_fingerprint: str | None = None
    parent_session_id: str | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: float = field(default_factory=_now)
    last_accessed_at: float = field(default_factory=_now)

    def set_system(self, text: str) -> None:
        msg = {"role": "system", "content": text}
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0] = msg
        else:
            self.messages.insert(0, msg)

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def derived_title(self) -> str:
        """A short label for the session list: the first user message, trimmed. Falls back
        to the id so a brand-new (empty) session still shows something."""
        for m in self.messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                text = " ".join(m["content"].split())
                if text:
                    return text[:48]
        return f"New chat · {self.id[:8]}"

    def summary(self) -> dict:
        """Lightweight metadata for the session list (no full message bodies)."""
        turns = sum(1 for m in self.messages if m.get("role") in ("user", "assistant"))
        return {
            "id": self.id,
            "title": self.title or self.derived_title(),
            "model": self.model,
            "message_count": turns,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
        }

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "messages": self.messages,
            "title": self.title,
            "system_prompt": self.system_prompt,
            "system_fingerprint": self.system_fingerprint,
            "parent_session_id": self.parent_session_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            model=data.get("model"),
            messages=data.get("messages") or [],
            title=data.get("title"),
            system_prompt=data.get("system_prompt"),
            system_fingerprint=data.get("system_fingerprint"),
            parent_session_id=data.get("parent_session_id"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            created_at=data.get("created_at") or _now(),
            last_accessed_at=data.get("last_accessed_at") or _now(),
        )


class SessionStore:
    """Durable session registry: an in-memory cache backed by file-per-session JSON.

    Persistence is the spring1 S1 keystone — conversations (GUI and Telegram) survive a
    backend restart instead of vanishing. The interface stays tiny (get / get_or_create /
    create / checkpoint / list / delete) so a SQLite/FTS backend can drop in later. Built
    without a directory it's a pure in-memory store (used by tests and any caller that
    doesn't want persistence).
    """

    def __init__(self, sessions_dir: Path | str | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._dir = Path(sessions_dir) if sessions_dir else None
        if self._dir is not None:
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._dir = None  # degrade to memory-only rather than crash startup

    # --- read ---

    def get(self, session_id: str) -> Session | None:
        """Return a session (loading from disk on a cache miss) without creating one."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        loaded = self._load(session_id)
        if loaded is not None:
            self._sessions[loaded.id] = loaded
        return loaded

    def get_or_create(
        self, session_id: str | None = None, model: str | None = None
    ) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                if model:
                    existing.model = model
                existing.last_accessed_at = _now()
                return existing
        sid = session_id or uuid.uuid4().hex
        session = Session(id=sid, model=model)
        self._sessions[sid] = session
        return session

    def list_sessions(self) -> list[dict]:
        """All session summaries, most-recently-used first. Merges the in-memory cache
        (freshest) with on-disk sessions not currently loaded."""
        summaries: dict[str, dict] = {s.id: s.summary() for s in self._sessions.values()}
        if self._dir is not None:
            for path in self._dir.glob("*.json"):
                sid = path.stem
                if sid in summaries:
                    continue
                data = self._read_json(path)
                if data is not None:
                    summaries[sid] = Session.from_dict(data).summary()
        return sorted(
            summaries.values(), key=lambda d: d["last_accessed_at"], reverse=True
        )

    # --- write ---

    def create(self, model: str | None = None) -> Session:
        """A fresh, empty session, persisted immediately so it appears in the list."""
        session = Session(id=uuid.uuid4().hex, model=model)
        self._sessions[session.id] = session
        self.checkpoint(session)
        return session

    def checkpoint(self, session: Session) -> None:
        """Persist a session after a turn. Atomic (tmp + ``os.replace``) so a crash
        mid-write never corrupts an existing session file."""
        session.last_accessed_at = _now()
        self._sessions[session.id] = session
        if self._dir is None:
            return
        path = self._dir / f"{session.id}.json"
        tmp = path.with_name(f"{session.id}.json.tmp")
        try:
            tmp.write_text(
                json.dumps(session.to_dict(), ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, path)
        except OSError:
            # Persistence is best-effort: a write failure must not break the live turn.
            tmp.unlink(missing_ok=True)

    def delete_session(self, session_id: str) -> bool:
        existed = self._sessions.pop(session_id, None) is not None
        if self._dir is not None:
            path = self._dir / f"{session_id}.json"
            if path.exists():
                path.unlink(missing_ok=True)
                existed = True
        return existed

    # --- internals ---

    def _load(self, session_id: str) -> Session | None:
        if self._dir is None:
            return None
        data = self._read_json(self._dir / f"{session_id}.json")
        return Session.from_dict(data) if data is not None else None

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None  # corrupt or missing → treat as absent, never crash
