"""Route-level checks for /models load/unload (model ids contain slashes: org/name)."""

from __future__ import annotations

import json

from assistant.config import Settings
from assistant.main import create_app
from assistant.models.mlx_service import MlxModelService
from fastapi.testclient import TestClient


def _make_model(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"\x00")
    return d


def test_load_route_accepts_slashed_model_id(tmp_path):
    # org/name ids must reach the handler — a plain `{model_id}` path param would 404 on
    # the slash. We expect the load to *fail* (no such model) but NOT a routing 404.
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        r = client.post("/models/mlx-community/Does-Not-Exist-4bit/load")
    assert r.status_code != 404, r.text
    assert r.status_code == 502  # reached the handler; load failed cleanly


def test_unload_route_accepts_slashed_model_id(tmp_path):
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        r = client.post("/models/mlx-community/Does-Not-Exist-4bit/unload")
    assert r.status_code != 404, r.text


def test_delete_removes_local_model_from_disk(tmp_path, monkeypatch):
    # The catalog fails soft to [] when mlx_lm isn't installed (the available() gate), so
    # on CI — dev deps only, no mlx — the list comes back empty and the delete can't be
    # exercised. This test is about route/catalog/file behavior, not the engine: force
    # availability on so it runs the same everywhere.
    monkeypatch.setattr(MlxModelService, "available", lambda self: True)
    model = _make_model(tmp_path / "qwen3-8b")
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        before = {m["id"] for m in client.get("/models").json()["models"]}
        r = client.delete("/models/qwen3-8b")
        after = {m["id"] for m in client.get("/models").json()["models"]}
    assert "qwen3-8b" in before
    assert r.status_code == 200, r.text
    assert not model.exists()  # files actually removed
    assert "qwen3-8b" not in after


def test_delete_unknown_model_is_400_not_500(tmp_path):
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        r = client.delete("/models/nope")
    assert r.status_code == 400, r.text


def test_models_list_flags_weak_at_tools(tmp_path, monkeypatch):
    # The GUI picker renders ⚠️ from this per-model flag (one source of truth with Telegram's
    # /models picker). A thinking model is flagged; its Coder sibling is not.
    # Availability forced on for the same reason as the delete test above: the flag logic
    # is pure name heuristics and must be testable without an mlx install.
    monkeypatch.setattr(MlxModelService, "available", lambda self: True)
    _make_model(tmp_path / "Qwen3-30B-A3B-8bit")
    _make_model(tmp_path / "Qwen3-Coder-30B-A3B-Instruct-8bit")
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        models = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert models["Qwen3-30B-A3B-8bit"]["weak_at_tools"] is True
    assert models["Qwen3-Coder-30B-A3B-Instruct-8bit"]["weak_at_tools"] is False


def test_models_route_filters_unloadable_and_flags_chattable(tmp_path, monkeypatch):
    # N103: /models must list only what an MLX engine can load, and each entry must carry the
    # backend's own `chattable` verdict so the GUI picker and the Telegram picker filter from ONE
    # definition instead of each keeping a type list that drifts (the GUI's let video in).
    monkeypatch.setattr(MlxModelService, "available", lambda self: True)
    _make_model(tmp_path / "chat-model")
    # An ASR checkpoint: listed (Models screen shows it) but never selectable as a chat model.
    asr = tmp_path / "Breeze-ASR-26-mlx"
    asr.mkdir()
    asr_cfg = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
    (asr / "config.json").write_text(json.dumps(asr_cfg))
    (asr / "model.safetensors").write_bytes(b"\x00")
    # A mid-training dump: no engine loads it, so it must not be listed at all.
    trained = _make_model(tmp_path / "half-trained")
    (trained / "optimizer.bin").write_bytes(b"\x00")

    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        body = client.get("/models").json()
    by_id = {m["id"]: m for m in body["models"]}

    assert "half-trained" not in by_id  # unloadable -> not offered anywhere
    assert by_id["chat-model"]["chattable"] is True
    assert by_id["Breeze-ASR-26-mlx"]["type"] == "audio"
    assert by_id["Breeze-ASR-26-mlx"]["chattable"] is False
