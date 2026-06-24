"""HTTP surface for the manual /compact endpoint (S6). The CompactionManager itself is unit
-tested in test_compaction.py; here we pin the endpoint's branches (404/400/delegate)."""

from __future__ import annotations

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path):
    return TestClient(
        create_app(
            Settings(sessions_dir=tmp_path / "sessions", models_dir=tmp_path / "models")
        )
    )


class FakeCompaction:
    def __init__(self, event):
        self._event = event

    async def force_compact(self, session, model):
        return self._event


def test_compact_unknown_session_is_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.post("/sessions/nope/compact").status_code == 404


def test_compact_requires_a_model(tmp_path):
    with _client(tmp_path) as client:
        sid = client.post("/sessions", json={}).json()["id"]  # created without a model
        assert client.post(f"/sessions/{sid}/compact").status_code == 400


def test_compact_happy_path_reports_event(tmp_path):
    event = {
        "type": "compaction",
        "dropped": 2,
        "tokens_before": 100,
        "tokens_after": 40,
        "context_window": 8192,
    }
    with _client(tmp_path) as client:
        client.app.state.compaction = FakeCompaction(event)
        sid = client.post("/sessions", json={"model": "m"}).json()["id"]
        body = client.post(f"/sessions/{sid}/compact").json()
        assert body["compacted"] is True and body["dropped"] == 2


def test_compact_nothing_old_enough_reports_no_change(tmp_path):
    with _client(tmp_path) as client:
        client.app.state.compaction = FakeCompaction(None)  # nothing to summarize
        sid = client.post("/sessions", json={"model": "m"}).json()["id"]
        body = client.post(f"/sessions/{sid}/compact").json()
        assert body["compacted"] is False and "context_tokens" in body
