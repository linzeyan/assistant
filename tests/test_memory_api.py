"""CRUD over /memory (GUI-driven memory management)."""

from __future__ import annotations

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path):
    return TestClient(create_app(Settings(memory_dir=tmp_path / "memory")))


def test_create_list_update_delete_cycle(tmp_path):
    with _client(tmp_path) as client:
        created = client.post(
            "/memory", json={"content": "Ricky prefers tabs", "tags": ["pref"]}
        )
        assert created.status_code == 200, created.text
        entry_id = created.json()["id"]

        listed = client.get("/memory").json()["memories"]
        assert any(m["id"] == entry_id and m["content"] == "Ricky prefers tabs" for m in listed)

        upd = client.put(
            f"/memory/{entry_id}", json={"content": "Ricky prefers spaces", "tags": ["pref", "style"]}
        )
        assert upd.status_code == 200
        assert upd.json()["content"] == "Ricky prefers spaces"
        assert "embedding" not in upd.json()  # internal field stays hidden

        assert client.delete(f"/memory/{entry_id}").status_code == 200
        assert all(m["id"] != entry_id for m in client.get("/memory").json()["memories"])


def test_create_requires_content(tmp_path):
    with _client(tmp_path) as client:
        assert client.post("/memory", json={"content": "  "}).status_code == 400


def test_update_unknown_id_returns_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.put("/memory/nope", json={"content": "x"}).status_code == 404


def test_delete_unknown_id_returns_404(tmp_path):
    with _client(tmp_path) as client:
        assert client.delete("/memory/nope").status_code == 404
