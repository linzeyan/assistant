"""POST /logs/clear — Settings "clear logs" truncates the backend log files."""

from __future__ import annotations

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def test_clear_logs_truncates_the_log_files(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "backend.log").write_text("noise\n" * 100)
    (log_dir / "backend.out.log").write_text("spawn tee\n")
    # app.log is written by the Swift app process itself (N62 boot-timing log), not the backend —
    # it lives in the same log_dir and must be cleared too, or "clear logs" leaves it behind.
    (log_dir / "app.log").write_text("app: launch\n")

    app = create_app(
        Settings(sessions_dir=tmp_path / "s", models_dir=tmp_path / "m", log_dir=log_dir)
    )
    with TestClient(app) as client:
        r = client.post("/logs/clear")

    assert r.status_code == 200
    cleared = r.json()["cleared"]
    assert "backend.log" in cleared and "backend.out.log" in cleared and "app.log" in cleared
    assert (log_dir / "backend.log").read_text() == ""  # truncated in place
    assert (log_dir / "backend.out.log").read_text() == ""
    assert (log_dir / "app.log").read_text() == ""


def test_clear_logs_is_ok_when_no_logs_exist(tmp_path):
    # A fresh install with no log files yet must not error — just report nothing cleared.
    app = create_app(
        Settings(sessions_dir=tmp_path / "s", models_dir=tmp_path / "m", log_dir=tmp_path / "logs")
    )
    with TestClient(app) as client:
        r = client.post("/logs/clear")
    assert r.status_code == 200 and r.json()["cleared"] == []
