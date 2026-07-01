"""Route-level checks for /models load/unload (model ids contain slashes: org/name)."""

from __future__ import annotations

import json

from assistant.config import Settings
from assistant.main import create_app
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


def test_delete_removes_local_model_from_disk(tmp_path):
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


def test_models_list_flags_weak_at_tools(tmp_path):
    # The GUI picker renders ⚠️ from this per-model flag (one source of truth with Telegram's
    # /models picker). A thinking model is flagged; its Coder sibling is not.
    _make_model(tmp_path / "Qwen3-30B-A3B-8bit")
    _make_model(tmp_path / "Qwen3-Coder-30B-A3B-Instruct-8bit")
    with TestClient(create_app(Settings(models_dir=tmp_path))) as client:
        models = {m["id"]: m for m in client.get("/models").json()["models"]}
    assert models["Qwen3-30B-A3B-8bit"]["weak_at_tools"] is True
    assert models["Qwen3-Coder-30B-A3B-Instruct-8bit"]["weak_at_tools"] is False
