from __future__ import annotations

import json
from pathlib import Path

from assistant.models.mlx_discovery import (
    DiscoveredModel,
    classify_kind,
    is_mlx_loadable,
    discover_hf_cache,
    discover_image_checkpoints,
    discover_local,
    discover_models,
    discover_video_checkpoints,
    is_video_checkpoint,
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


def test_classify_kind_omni_with_vae_is_not_chattable(tmp_path):
    # Lance-3B-Video declares model_type qwen2_5_vl (would read as vlm = chattable) but ships a
    # diffusion vae.safetensors — it's an omni gen model mlx-vlm can't load. The vae is the
    # signal to keep it OUT of the chat picker (N32); without it, picking it dumped 1606 weights.
    d = tmp_path / "m"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2_5_VLForConditionalGeneration"],
                    "model_type": "qwen2_5_vl", "vision_config": {}})
    )
    (d / "vae.safetensors").write_bytes(b"\x00")
    assert classify_kind(d) == "video"  # not "vlm" → excluded from the chat list


def test_classify_kind_image_via_pipeline_class(tmp_path):
    # A FLUX/Qwen-Image diffusers checkpoint declares its pipeline in model_index.json and
    # also ships a VAE — without the image branch it would read as "video" (the vae rule) or
    # "llm". It belongs under the Models "Image" tab.
    d = tmp_path / "flux"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    (d / "vae.safetensors").write_bytes(b"\x00")  # image checkpoints carry a VAE too
    assert classify_kind(d) == "image"


def test_classify_kind_qwen_image_via_arch(tmp_path):
    d = tmp_path / "qi"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["QwenImageTransformer2DModel"]}))
    assert classify_kind(d) == "image"


def test_classify_kind_video_via_model_index(tmp_path):
    # A diffusers Wan pipeline declares its class in model_index.json (no config.json). Without
    # folding that in it would fail-open to "llm" and pollute the chat list.
    d = tmp_path / "wanp"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline"}))
    assert classify_kind(d) == "video"


def test_discover_includes_diffusers_without_config_json(tmp_path):
    # A FLUX checkpoint carries model_index.json (not config.json) + weights in subdirs. It must
    # not silently vanish from the Models list just because it lacks a top-level config.json.
    from assistant.models.mlx_discovery import discover_local

    d = tmp_path / "FLUX.1-schnell"
    (d / "transformer").mkdir(parents=True)
    (d / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    (d / "transformer" / "diffusion.safetensors").write_bytes(b"\x00" * 16)
    found = {m.id: m for m in discover_local(tmp_path)}
    assert "FLUX.1-schnell" in found and found["FLUX.1-schnell"].kind == "image"


def _make_video_ckpt(d: Path, *, dual: bool = False, t5: bool = True, vae: bool = True) -> Path:
    """A converted-MLX Wan checkpoint: config + vae + t5_encoder + transformer weight(s)."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    if vae:
        (d / "vae.safetensors").write_bytes(b"\x00")
    if t5:
        (d / "t5_encoder.safetensors").write_bytes(b"\x00")
    if dual:
        (d / "low_noise_model.safetensors").write_bytes(b"\x00")
        (d / "high_noise_model.safetensors").write_bytes(b"\x00")
    else:
        (d / "model.safetensors").write_bytes(b"\x00")
    return d


def test_is_video_checkpoint_requires_converted_layout(tmp_path):
    assert is_video_checkpoint(_make_video_ckpt(tmp_path / "single"))
    assert is_video_checkpoint(_make_video_ckpt(tmp_path / "dual", dual=True))
    # Missing any required component → not loadable by mlx-video.
    assert not is_video_checkpoint(_make_video_ckpt(tmp_path / "no_t5", t5=False))
    assert not is_video_checkpoint(_make_video_ckpt(tmp_path / "no_vae", vae=False))


def test_smallest_of_kind_picks_fewest_bytes():
    from assistant.models.mlx_discovery import DiscoveredModel, smallest_of_kind

    models = [
        DiscoveredModel("big-llm", Path("/a"), "local", "llm", 100),
        DiscoveredModel("small-llm", Path("/b"), "local", "llm", 10),
        DiscoveredModel("img", Path("/c"), "local", "image", 5),
        DiscoveredModel("unknown-size", Path("/d"), "local", "llm", 0),
    ]
    assert smallest_of_kind(models, {"llm"}).id == "small-llm"
    assert smallest_of_kind(models, {"llm", "image"}).id == "img"  # cross-kind, fewest bytes wins
    assert smallest_of_kind(models, {"vlm"}) is None  # no model of that kind
    # A zero/unknown size means "not measured" — never picked as the smallest.
    assert smallest_of_kind([models[3]], {"llm"}) is None


def _make_component_pipeline(d: Path) -> Path:
    """An mlx-gen image checkpoint: diffusers components split into subdirs (transformer/ + vae/)
    with sharded weights and NO top-level config.json / model_index.json."""
    d.mkdir(parents=True, exist_ok=True)
    for comp in ("transformer", "vae", "text_encoder"):
        (d / comp).mkdir()
        (d / comp / "0.safetensors").write_bytes(b"\x00" * 16)
    return d


def test_classify_component_pipeline_is_image(tmp_path):
    # z-image-turbo / qwen-image-edit-2511: no config anywhere, only the transformer+vae dirs.
    # Without the structural signal it fail-opens to "llm" and never reaches the Image tab.
    d = _make_component_pipeline(tmp_path / "z-image-turbo-8bit")
    assert classify_kind(d) == "image"


def test_classify_component_pipeline_video_by_name(tmp_path):
    # The same component layout with a video-marked dir name routes to video, not image.
    d = _make_component_pipeline(tmp_path / "Wan2.2-t2v-component")
    assert classify_kind(d) == "video"


def test_discover_image_checkpoints_finds_component_models(tmp_path):
    org = tmp_path / "AbstractFramework"
    _make_component_pipeline(org / "z-image-turbo-8bit")
    _make_component_pipeline(org / "qwen-image-edit-2511-8bit")
    _make_model(tmp_path / "qwen3-8b")  # a chat model must NOT show up here
    found = {m.id for m in discover_image_checkpoints([tmp_path])}
    assert found == {
        "AbstractFramework/z-image-turbo-8bit",
        "AbstractFramework/qwen-image-edit-2511-8bit",
    }


def test_discover_video_checkpoints_excludes_raw_hf_dir(tmp_path):
    # A converted-MLX dir is loadable; a raw HF Wan download (model_type ti2v but only
    # diffusion_pytorch_model-* weights, no vae/t5_encoder) classifies as video yet mlx-video
    # cannot load it — the picker must offer only the converted one.
    primary = tmp_path / "primary"
    _make_video_ckpt(primary / "Wan2.2-TI2V-5B-mlx")
    raw = primary / "Wan2.2-TI2V-5B"
    raw.mkdir(parents=True)
    (raw / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    (raw / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(b"\x00")
    found = discover_video_checkpoints([primary])
    assert [m.id for m in found] == ["Wan2.2-TI2V-5B-mlx"]


def test_discover_video_checkpoints_dedupes_across_dirs(tmp_path):
    _make_video_ckpt(tmp_path / "a" / "Wan-mlx")
    _make_video_ckpt(tmp_path / "b" / "Wan-mlx")  # same id in a second dir
    found = discover_video_checkpoints([tmp_path / "a", tmp_path / "b"])
    assert [m.id for m in found] == ["Wan-mlx"]  # first dir wins, no duplicate


def test_classify_kind_whisper_is_audio_not_chat(tmp_path):
    # N103: Breeze-ASR ships `WhisperForConditionalGeneration` with no vision_config, so it used
    # to fail open to "llm" and appear in the CHAT picker — where mlx-lm cannot load it at all
    # (the Wan-video bug of N27, in a different family). It belongs to the audio backend.
    d = tmp_path / "Breeze-ASR-26-mlx"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]})
    )
    (d / "model.safetensors").write_bytes(b"\x00")
    assert classify_kind(d) == "audio"


def test_classify_kind_fish_tts_is_audio_not_chat(tmp_path):
    # mlx-community/fish-audio-s2-pro-bf16 declares model_type "fish_qwen3_omni" (mlx-audio's
    # Fish TTS family): a text LLM plus an audio decoder head, no architectures key at all — so
    # it failed open to "llm" and landed in the CHAT picker. It belongs to the audio backend.
    d = tmp_path / "fish-audio-s2-pro-bf16"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "fish_qwen3_omni", "audio_decoder_config": {"dim": 2560}})
    )
    (d / "model.safetensors").write_bytes(b"\x00")
    assert classify_kind(d) == "audio"


def test_classify_kind_mtp_drafter_is_draft_not_chat(tmp_path):
    # An MTP drafter (mlx-community/Qwen3.8-27B-MTP-8bit, model_type "qwen3_5_mtp") mirrors its
    # target's text_config, so it read as a plain "llm" and polluted the chat picker — but the
    # checkpoint holds only drafter weights (no embeddings/LM head) and is not standalone. It
    # pairs with a target via the per-model "draft" setting instead.
    d = tmp_path / "Qwen3.8-27B-MTP-8bit"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_mtp", "block_size": 3, "text_config": {"hidden_size": 5120}})
    )
    (d / "model.safetensors").write_bytes(b"\x00")
    assert classify_kind(d) == "draft"


def _discovered(path, kind="llm"):
    return DiscoveredModel(id=path.name, path=path, source="local", kind=kind, size_bytes=1)


def test_is_mlx_loadable_rejects_training_checkpoint(tmp_path):
    # A mid-training dump (observed: Breeze-ASR-25 carries optimizer.bin + scheduler.bin beside
    # real weights) is not a distributable model — no engine loads one, so it must not be listed.
    d = _make_model(tmp_path / "half-trained")
    (d / "optimizer.bin").write_bytes(b"\x00")
    assert is_mlx_loadable(_discovered(d)) is False


def test_is_mlx_loadable_rejects_unconvertible_video_dir(tmp_path):
    # Raw HF Wan downloads and chat-arch checkpoints carrying a diffusion VAE (Lance-3B-Video,
    # N32) load in NEITHER engine: mlx-video needs the converted layout, mlx-lm/vlm choke on the
    # weights. Listing them is the phantom-model failure this filter exists to stop.
    raw = tmp_path / "Wan2.2-TI2V-5B"
    raw.mkdir()
    (raw / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    (raw / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(b"\x00")
    assert is_mlx_loadable(_discovered(raw, kind="video")) is False


def test_is_mlx_loadable_keeps_plain_and_converted_models(tmp_path):
    # Structural, never name-based: a plain HuggingFace-named repo that mlx-lm loads fine must
    # stay listed — an "mlx-community/" prefix is not what makes a model loadable.
    plain = _make_model(tmp_path / "some-org--Qwen3-8B")
    assert is_mlx_loadable(_discovered(plain)) is True
    converted = _make_video_ckpt(tmp_path / "Wan2.2-TI2V-5B-mlx")
    assert is_mlx_loadable(_discovered(converted, kind="video")) is True


def test_is_mlx_loadable_rejects_pytorch_only_weights(tmp_path):
    d = tmp_path / "torch-only"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "pytorch_model.bin").write_bytes(b"\x00")  # every MLX loader reads safetensors
    assert is_mlx_loadable(_discovered(d)) is False
