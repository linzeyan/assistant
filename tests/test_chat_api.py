"""HTTP /chat behaviour. A turn that fails mid-stream must still persist the conversation
(#4) — the user's message shouldn't vanish from the session list just because the model
turn blew up (e.g. a chat-template render error)."""

from __future__ import annotations

import json

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path):
    return TestClient(
        create_app(
            Settings(sessions_dir=tmp_path / "sessions", models_dir=tmp_path / "models")
        )
    )


class _OkService:
    async def reachable(self):
        return True


class _BoomAgent:
    """Mirrors the real loop: the user message is recorded before the turn blows up."""

    async def run(self, session, message, model, approver=None):
        session.add_user(message)
        raise RuntimeError("template render boom")
        yield  # pragma: no cover - unreachable, makes this an async generator


def _session_id_from_sse(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("data:"):
            ev = json.loads(line[len("data:"):].strip())
            if ev.get("type") == "session":
                return ev["session_id"]
    return None


def test_failed_turn_is_still_checkpointed(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        client.app.state.agent = _BoomAgent()
        r = client.post("/chat", json={"message": "hi", "model": "m"})
        assert r.status_code == 200
        assert "error" in r.text  # the failure was streamed, not silently dropped
        sid = _session_id_from_sse(r.text)
        assert sid
        # Persisted to disk despite the failure (the #4 symptom was the file never landing).
        assert (tmp_path / "sessions" / f"{sid}.json").exists()
        reloaded = client.app.state.sessions.get(sid)
        assert any(
            m.get("role") == "user" and m["content"] == "hi" for m in reloaded.messages
        )
