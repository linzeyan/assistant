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


def test_search_endpoint_finds_session_and_returns_snippet(tmp_path):
    # F/S14: GET /sessions/search is matched as a literal path (declared before /{session_id}),
    # not swallowed by the dynamic route, and returns matching sessions with a snippet.
    with _client(tmp_path) as client:
        store = client.app.state.sessions
        s = store.create(model="m")
        s.add_user("Investigate the flaky telegram heartbeat")
        store.checkpoint(s)

        r = client.get("/sessions/search", params={"q": "heartbeat"})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "heartbeat" and body["count"] == 1
        assert body["results"][0]["id"] == s.id
        assert "heartbeat" in body["results"][0]["snippet"].lower()

        # A blank query is well-formed and simply returns nothing (not a 422, not everything).
        empty = client.get("/sessions/search", params={"q": ""})
        assert empty.status_code == 200 and empty.json()["count"] == 0


def test_search_path_not_shadowed_by_dynamic_session_route(tmp_path):
    # Guards the route-ordering gotcha: /sessions/search must not resolve to get_session("search").
    with _client(tmp_path) as client:
        r = client.get("/sessions/search", params={"q": "anything"})
        assert r.status_code == 200  # would be 404 ("unknown session: search") if shadowed
        assert "results" in r.json()
