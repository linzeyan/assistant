"""GET/PUT /config: whitelisted path editing, persisted to config.toml."""

from __future__ import annotations

import tomllib

import assistant.api.routes_config as routes_config
from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    # Redirect the module-level config path into the test's tmp dir.
    cfg = tmp_path / "config.toml"
    monkeypatch.setattr(routes_config, "_CONFIG_PATH", cfg)
    app = create_app(Settings(models_dir=tmp_path / "models"))
    return TestClient(app), cfg


def test_get_config_reports_paths(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        body = client.get("/config").json()
    assert body["models_dir"].endswith("/models")
    assert "download_dir" in body
    assert body["backend_host"] == "127.0.0.1"
    assert body["backend_port"] == 9981
    assert body["model_backend"] == "mlx"
    assert body["extra_model_dirs"] == []
    assert body["hf_cache"] is False
    assert body["max_output_tokens"] == 4096
    # Gateways (S9): status is surfaced here; the token is masked (None when unset).
    assert body["telegram_configured"] is False
    assert body["telegram_token_masked"] is None
    assert body["telegram_allowed_users"] == []


def test_put_config_sets_telegram_allowlist_live(tmp_path, monkeypatch):
    client, cfg = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"telegram_allowed_users": [111, 222]})
        body = client.get("/config").json()
    assert resp.status_code == 200 and resp.json()["restart_required"] is False
    assert body["telegram_allowed_users"] == [111, 222]
    assert tomllib.loads(cfg.read_text())["telegram_allowed_users"] == [111, 222]


def test_put_config_rejects_whitespace_telegram_token(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"telegram_token": "bad token"})
    assert resp.status_code == 400


def test_put_config_sets_max_output_tokens_live(tmp_path, monkeypatch):
    # The generation ceiling applies live (next turn reads the loop's value), so it persists
    # to config.toml AND updates the running AgentLoop without a restart.
    client, cfg = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"max_output_tokens": 8192})
        live = client.app.state.agent._max_output_tokens
        live_settings = client.app.state.settings.max_output_tokens
    assert resp.status_code == 200 and resp.json()["restart_required"] is False
    assert live == 8192 and live_settings == 8192
    assert tomllib.loads(cfg.read_text())["max_output_tokens"] == 8192


def test_put_config_rejects_out_of_range_max_output_tokens(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        too_low = client.put("/config", json={"max_output_tokens": 16})
        too_high = client.put("/config", json={"max_output_tokens": 999999})
    assert too_low.status_code == 400 and too_high.status_code == 400


def test_put_config_writes_extra_model_dirs_and_hf_cache(tmp_path, monkeypatch):
    client, cfg = _client(tmp_path, monkeypatch)
    extra = str(tmp_path / "more-models")
    with client:
        ok = client.put("/config", json={"extra_model_dirs": [extra], "hf_cache": True})
        bad = client.put("/config", json={"extra_model_dirs": ["relative/dir"]})
    assert ok.status_code == 200
    assert bad.status_code == 400
    written = tomllib.loads(cfg.read_text())
    assert written["extra_model_dirs"] == [extra]
    assert written["hf_cache"] is True


def test_put_config_switches_model_backend(tmp_path, monkeypatch):
    client, cfg = _client(tmp_path, monkeypatch)
    with client:
        ok = client.put("/config", json={"model_backend": "omlx"})
        bad = client.put("/config", json={"model_backend": "bogus"})
    assert ok.status_code == 200
    assert bad.status_code == 400
    assert tomllib.loads(cfg.read_text())["model_backend"] == "omlx"


def test_put_config_writes_host_and_port(tmp_path, monkeypatch):
    client, cfg = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"backend_host": "0.0.0.0", "backend_port": 9000})
    assert resp.status_code == 200
    written = tomllib.loads(cfg.read_text())
    assert written["backend_host"] == "0.0.0.0"
    assert written["backend_port"] == 9000


def test_put_config_rejects_bad_port(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"backend_port": 70000})
    assert resp.status_code == 400


def test_put_config_writes_toml_and_applies_live(tmp_path, monkeypatch):
    # Model-dir changes are discovery-only: applied to the running service immediately,
    # so they persist to config.toml AND don't require a restart.
    client, cfg = _client(tmp_path, monkeypatch)
    new_dir = str(tmp_path / "weights")
    with client:
        resp = client.put("/config", json={"models_dir": new_dir})
        live = client.app.state.model_service._models_dir
        live_settings = client.app.state.settings.models_dir
    body = resp.json()
    assert body["restart_required"] is False
    assert body["updated"]["models_dir"] == new_dir
    assert str(live) == new_dir  # service re-pointed without a restart
    assert str(live_settings) == new_dir
    written = tomllib.loads(cfg.read_text())
    assert written["models_dir"] == new_dir


def test_put_config_host_change_requires_restart(tmp_path, monkeypatch):
    # Rebinding the socket can't happen in place, so host/port stay restart-gated.
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"backend_port": 9000})
    assert resp.json()["restart_required"] is True


def test_put_config_rejects_relative_path(tmp_path, monkeypatch):
    client, _ = _client(tmp_path, monkeypatch)
    with client:
        resp = client.put("/config", json={"models_dir": "relative/dir"})
    assert resp.status_code == 400


def test_put_config_merges_existing_keys(tmp_path, monkeypatch):
    client, cfg = _client(tmp_path, monkeypatch)
    cfg.write_text('approval_required = false\n')
    with client:
        client.put("/config", json={"download_dir": str(tmp_path / "dl")})
    written = tomllib.loads(cfg.read_text())
    assert written["approval_required"] is False  # preserved
    assert written["download_dir"].endswith("/dl")
