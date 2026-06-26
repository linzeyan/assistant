from __future__ import annotations

import json
from pathlib import Path

from assistant.models.mlx_discovery import (
    classify_kind,
    discover_hf_cache,
    discover_local,
    discover_models,
)


def _make_model(d: Path, *, arch: str = "LlamaForCausalLM", weights: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"architectures": [arch]}))
    if weights:  # a usable model carries real weights, not just config metadata
        (d / "model.safetensors").write_bytes(b"\x00")
    return d


def test_discover_local_flat(tmp_path):
    _make_model(tmp_path / "qwen3-8b")
    (tmp_path / "not-a-model").mkdir()  # no config.json -> ignored
    found = discover_local(tmp_path)
    assert [m.id for m in found] == ["qwen3-8b"]
    assert found[0].source == "local"
    assert found[0].path == tmp_path / "qwen3-8b"


def test_discover_reports_on_disk_size(tmp_path):
    d = _make_model(tmp_path / "qwen3-8b")
    (d / "model.safetensors").write_bytes(b"\x00" * 2048)  # overwrite the 1-byte stub
    found = discover_local(tmp_path)
    # config.json + the 2048-byte weight → size reflects the real footprint.
    assert found[0].size_bytes >= 2048


def test_discover_local_two_level_org(tmp_path):
    _make_model(tmp_path / "mlx-community" / "Llama-3-8B-Instruct")
    found = discover_local(tmp_path)
    assert [m.id for m in found] == ["mlx-community/Llama-3-8B-Instruct"]


def test_discover_hf_cache(tmp_path):
    snap = tmp_path / "models--mlx-community--Qwen3-8B" / "snapshots" / "abc123"
    _make_model(snap)
    found = discover_hf_cache(tmp_path)
    assert found[0].id == "mlx-community/Qwen3-8B"
    assert found[0].source == "hf_cache"
    assert found[0].path == snap


def test_discover_hf_cache_skips_repos_without_config(tmp_path):
    # A repo whose snapshot has no config.json is not a usable model.
    (tmp_path / "models--x--incomplete" / "snapshots" / "h").mkdir(parents=True)
    assert discover_hf_cache(tmp_path) == []


def test_discover_models_local_shadows_cache(tmp_path):
    models = _make_model(tmp_path / "models" / "shared").parent
    cache = tmp_path / "cache"
    _make_model(cache / "models--org--shared" / "snapshots" / "h")
    # Local "shared" and cache "org/shared" have different ids, so both surface.
    found = discover_models(models, include_hf_cache=True, hf_cache_dir=cache)
    ids = {m.id for m in found}
    assert ids == {"shared", "org/shared"}
    # include_hf_cache=False keeps only local.
    assert [m.id for m in discover_models(models, include_hf_cache=False)] == ["shared"]


def test_discover_models_scans_extra_dirs(tmp_path):
    primary = _make_model(tmp_path / "primary" / "a").parent
    extra = _make_model(tmp_path / "extra" / "b").parent
    # The cache is NOT scanned by default — only the configured dirs surface.
    _make_model(tmp_path / "cache" / "models--org--c" / "snapshots" / "h")
    found = discover_models(primary, extra_dirs=[extra], hf_cache_dir=tmp_path / "cache")
    assert {m.id for m in found} == {"a", "b"}  # cache "org/c" excluded


def test_discover_models_primary_shadows_extra_on_id_collision(tmp_path):
    primary = _make_model(tmp_path / "primary" / "dup").parent
    extra = _make_model(tmp_path / "extra" / "dup").parent
    found = discover_models(primary, extra_dirs=[extra])
    assert [m.id for m in found] == ["dup"]
    assert found[0].path == tmp_path / "primary" / "dup"  # primary wins


def test_skips_metadata_only_entry_without_weights(tmp_path):
    # config.json but no weight files = a metadata-only cache entry (HF fetched config,
    # or an interrupted download). It must not appear as a loadable model.
    _make_model(tmp_path / "local" / "phantom", weights=False)
    assert discover_local(tmp_path / "local") == []
    snap = tmp_path / "cache" / "models--org--phantom" / "snapshots" / "h"
    _make_model(snap, weights=False)
    assert discover_hf_cache(tmp_path / "cache") == []


def test_discover_tags_model_kind(tmp_path):
    _make_model(tmp_path / "chat", arch="Qwen2ForCausalLM")
    _make_model(tmp_path / "vision", arch="Qwen2_5_VLForConditionalGeneration")
    _make_model(tmp_path / "encoder", arch="UMT5EncoderModel")
    kinds = {m.id: m.kind for m in discover_local(tmp_path)}
    assert kinds == {"chat": "llm", "vision": "vlm", "encoder": "embedding"}


def test_classify_kind_vlm_via_vision_config(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    # Some VLMs carry a causal-LM arch; the nested vision_config is the giveaway.
    (d / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "vision_config": {}})
    )
    assert classify_kind(d) == "vlm"


def test_classify_kind_defaults_to_llm_when_unknown(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text("{}")  # unrecognised → fail open, don't block load
    assert classify_kind(d) == "llm"


def test_classify_kind_video_via_model_type(tmp_path):
    # Wan TI2V (text/image-to-video) advertises a video model_type. Without "video" it
    # fail-opens to "llm" and pollutes the chat list, where loading it throws.
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    assert classify_kind(d) == "video"


def test_classify_kind_video_via_wan_class_name(tmp_path):
    # Wan S2V's config has no model_type/architectures, only a _class_name. The diffusers
    # class is the giveaway that this is generative video, not a chat model.
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"_class_name": "WanModel_S2V"}))
    assert classify_kind(d) == "video"
