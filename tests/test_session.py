from __future__ import annotations

from assistant.agent.session import SessionStore


def test_in_memory_get_or_create_roundtrip():
    store = SessionStore()
    s = store.get_or_create(model="m")
    s.add_user("hello")
    assert store.get_or_create(s.id).messages == s.messages


def test_checkpoint_persists_across_restart(tmp_path):
    store = SessionStore(tmp_path)
    s = store.get_or_create(model="qwen")
    s.add_user("remember me")
    s.add_assistant("ok")
    store.checkpoint(s)

    # Simulate a backend restart: a fresh store over the same dir must reload the session.
    reloaded = SessionStore(tmp_path).get(s.id)
    assert reloaded is not None
    assert reloaded.model == "qwen"
    assert [m["content"] for m in reloaded.messages] == ["remember me", "ok"]


def test_checkpoint_is_atomic_no_tmp_left(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create(model="m")
    s.add_user("x")
    store.checkpoint(s)
    assert not list(tmp_path.glob("*.tmp"))  # tmp file replaced, not orphaned
    assert (tmp_path / f"{s.id}.json").is_file()


def test_list_sessions_orders_by_recency_with_titles(tmp_path):
    store = SessionStore(tmp_path)
    a = store.create(model="m")
    a.add_user("first question")
    b = store.create(model="m")
    b.add_user("second question")
    # Pin recency deterministically (avoid relying on wall-clock resolution).
    a.last_accessed_at = 1.0
    b.last_accessed_at = 2.0
    rows = store.list_sessions()
    assert [r["id"] for r in rows] == [b.id, a.id]
    assert {r["id"]: r["title"] for r in rows}[a.id] == "first question"


def test_list_includes_unloaded_disk_sessions(tmp_path):
    SessionStore(tmp_path).create(model="m")  # persisted, then that store is dropped
    fresh = SessionStore(tmp_path)  # nothing in memory yet
    assert len(fresh.list_sessions()) == 1


def test_delete_removes_memory_and_disk(tmp_path):
    store = SessionStore(tmp_path)
    s = store.create(model="m")
    assert (tmp_path / f"{s.id}.json").is_file()
    assert store.delete_session(s.id) is True
    assert not (tmp_path / f"{s.id}.json").exists()
    assert store.get(s.id) is None
    assert store.delete_session("nope") is False


def test_corrupt_session_file_is_skipped(tmp_path):
    (tmp_path / "bad.json").write_text("{ not json")
    store = SessionStore(tmp_path)
    assert store.get("bad") is None  # tolerated, not raised
    assert store.list_sessions() == []
