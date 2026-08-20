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

    async def run(
        self, session, message, model, approver=None, cwd=None, max_iters=None,
        template_kwargs=None,
    ):
        session.add_user(message)
        raise RuntimeError("template render boom")
        yield  # pragma: no cover - unreachable, makes this an async generator


class _RecordingAgent:
    """Records what the route handed the loop (cwd, per-turn template kwargs), then ends the
    turn immediately."""

    def __init__(self):
        self.cwd = "unset"
        self.template_kwargs = "unset"

    async def run(
        self, session, message, model, approver=None, cwd=None, max_iters=None,
        template_kwargs=None,
    ):
        self.cwd = cwd
        self.template_kwargs = template_kwargs
        session.add_user(message)
        yield {"type": "done", "usage": {}}


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


def test_workspace_overrides_the_server_default(tmp_path):
    """One backend has to serve several checkouts at once — a client driving N git worktrees
    keeps them apart only if the turn's cwd comes from the request, not from startup config."""
    tree = tmp_path / "worktree"
    tree.mkdir()
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        agent = client.app.state.agent = _RecordingAgent()
        r = client.post(
            "/chat", json={"message": "hi", "model": "m", "workspace": str(tree)}
        )
        assert r.status_code == 200
        assert agent.cwd == str(tree.resolve())


def test_workspace_defaults_to_none_when_omitted(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        agent = client.app.state.agent = _RecordingAgent()
        assert client.post("/chat", json={"message": "hi", "model": "m"}).status_code == 200
        assert agent.cwd is None  # the loop falls back to the configured workspace_dir


def test_per_turn_reasoning_knobs_reach_the_loop(tmp_path):
    """The Chat header's Thinking/Effort menu is per-conversation: it rides the request, so
    flipping it must not require saving anything against the model."""
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        agent = client.app.state.agent = _RecordingAgent()
        r = client.post(
            "/chat",
            json={"message": "hi", "model": "m", "thinking": False, "reasoning_effort": "low"},
        )
        assert r.status_code == 200
        assert agent.template_kwargs == {"enable_thinking": False, "reasoning_effort": "low"}


def test_unset_reasoning_knobs_send_nothing(tmp_path):
    """"Model default" has to mean *absent*: pinning enable_thinking=True here would override
    a model whose saved setting says otherwise."""
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        agent = client.app.state.agent = _RecordingAgent()
        assert client.post("/chat", json={"message": "hi", "model": "m"}).status_code == 200
        assert agent.template_kwargs == {}


def test_workspace_that_is_not_a_directory_is_rejected(tmp_path):
    """Rejecting up front beats letting the agent burn a turn discovering the path is wrong."""
    with _client(tmp_path) as client:
        client.app.state.model_service = _OkService()
        client.app.state.agent = _RecordingAgent()
        r = client.post(
            "/chat", json={"message": "hi", "model": "m", "workspace": str(tmp_path / "nope")}
        )
        assert r.status_code == 400
