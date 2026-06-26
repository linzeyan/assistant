"""Local MLX model discovery (native backend).

Mirrors omlx's ``model_discovery``: a model is any directory holding a
``config.json``. We scan two sources:

1. A local models directory, supporting both a flat layout (``<dir>/<model>``) and
   the HuggingFace two-level org layout (``<dir>/<org>/<model>``).
2. The HuggingFace hub cache (``~/.cache/huggingface/hub/models--<org>--<name>``),
   which is where ``mlx-lm`` / ``mflux`` download weights by default.

Discovery is pure filesystem inspection — no model is loaded — so it is cheap and
safe to call on every ``/models`` request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass(frozen=True)
class DiscoveredModel:
    id: str  # stable identifier the rest of the app uses to load/switch
    path: Path  # directory passed to ``mlx_lm.load``
    source: str  # "local" | "hf_cache"
    kind: str = "llm"  # "llm" | "vlm" | "embedding" — which engine can load it
    size_bytes: int = 0  # on-disk footprint, so the GUI can show / manage capacity


def _has_config(d: Path) -> bool:
    return (d / "config.json").is_file()


def _dir_size(d: Path) -> int:
    """On-disk size of a model dir. HF snapshots symlink into blobs/; ``is_file()``
    follows the link so the real weight size is counted, not the link's."""
    total = 0
    for f in d.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass  # broken symlink / race — skip, never fail discovery
    return total


# Weight files an mlx / transformers model carries. A snapshot with config.json but
# no resolvable weights is a metadata-only cache entry — HF fetched config to inspect
# the repo, or a download was interrupted — not a loadable model. Listing those puts
# phantom models in the picker, so a model must also carry real weights.
_WEIGHT_SUFFIXES = (".safetensors", ".npz", ".gguf", ".bin")


def _has_weights(d: Path) -> bool:
    # HF snapshots are symlinks into blobs/; `is_file()` follows the link, so an
    # unmaterialised weight (broken symlink) correctly counts as absent.
    for f in d.rglob("*"):
        if f.suffix in _WEIGHT_SUFFIXES and f.is_file() and f.stat().st_size > 0:
            return True
    return False


def _is_model(d: Path) -> bool:
    return _has_config(d) and _has_weights(d)


# Model-type substrings (HF ``model_type``) that are vision-language, plus text-only
# encoder/embedding families. Used to route a load to the right engine: mlx-lm loads
# text LLMs only, so handing it a VLM (mlx-vlm's job) or a diffusion text-encoder
# errors out. Classification is fail-open — an unrecognised arch defaults to "llm" so
# we never block a genuinely loadable model on an overzealous guess.
_VLM_TYPES = (
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "llava", "idefics", "paligemma",
    "mllama", "internvl", "pixtral", "smolvlm", "phi3_v", "got_ocr2", "aria",
)
_ENCODER_TYPES = {
    "t5", "umt5", "mt5", "bert", "roberta", "xlm-roberta", "clip", "siglip",
}
# Diffusion video model_type tags (Wan family: sound/text/image-to-video). These are
# generative video, not chat — see classify_kind for why misclassifying them matters.
_VIDEO_TYPES = {"s2v", "ti2v", "t2v", "i2v"}


def _read_config(d: Path) -> dict:
    try:
        return json.loads((d / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def classify_kind(d: Path) -> str:
    cfg = _read_config(d)
    archs = cfg.get("architectures") or []
    arch = (archs[0] if archs else "").lower()
    mtype = str(cfg.get("model_type") or "").lower()
    cls_name = str(cfg.get("_class_name") or "").lower()
    # Diffusion video models (Wan S2V/TI2V/T2V/I2V …) declare a video model_type or a
    # WanModel diffusers class. They are NOT chattable: without this they fail-open to
    # "llm" below and pollute the chat model list, where loading one as an LLM dies with
    # "No safetensors found" (weights are diffusion_pytorch_model-*, not LM format).
    if mtype in _VIDEO_TYPES or "wan" in cls_name or "video" in cls_name:
        return "video"
    # Vision-language first: a VL config also carries a causal-LM head, so it would
    # otherwise read as a plain LLM. ``vision_config`` is the strongest signal.
    if (
        "vision_config" in cfg
        or "image_token_id" in cfg
        or "vision" in arch
        or any(t in mtype or t in arch for t in _VLM_TYPES)
    ):
        return "vlm"
    # Text encoders / embeddings (T5/UMT5/BERT/CLIP …) — no generative chat. These
    # surface because diffusion stacks (mflux) cache their text encoders alongside.
    if arch.endswith("encodermodel") or arch.endswith("model") or mtype in _ENCODER_TYPES:
        return "embedding"
    return "llm"


def discover_local(models_dir: Path) -> list[DiscoveredModel]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    found: list[DiscoveredModel] = []
    for child in sorted(p for p in models_dir.iterdir() if p.is_dir()):
        if _is_model(child):
            found.append(
                DiscoveredModel(
                    child.name, child, "local", classify_kind(child), _dir_size(child)
                )
            )
            continue
        # Two-level org layout: <dir>/<org>/<model>/config.json
        for sub in sorted(p for p in child.iterdir() if p.is_dir()):
            if _is_model(sub):
                found.append(
                    DiscoveredModel(
                        f"{child.name}/{sub.name}", sub, "local",
                        classify_kind(sub), _dir_size(sub),
                    )
                )
    return found


def discover_hf_cache(cache_dir: Path | None = None) -> list[DiscoveredModel]:
    cache = Path(cache_dir) if cache_dir is not None else _HF_CACHE
    if not cache.is_dir():
        return []
    found: list[DiscoveredModel] = []
    for repo in sorted(cache.glob("models--*")):
        # models--<org>--<name>  ->  <org>/<name>
        name = repo.name[len("models--") :].replace("--", "/")
        snapshots = repo / "snapshots"
        if not snapshots.is_dir():
            continue
        snaps = [s for s in snapshots.iterdir() if s.is_dir() and _is_model(s)]
        if not snaps:
            continue
        # Prefer the most recently materialised snapshot.
        snap = max(snaps, key=lambda p: p.stat().st_mtime)
        found.append(
            DiscoveredModel(name, snap, "hf_cache", classify_kind(snap), _dir_size(snap))
        )
    return found


def discover_models(
    models_dir: Path,
    include_hf_cache: bool = False,
    hf_cache_dir: Path | None = None,
    extra_dirs: list[Path] | None = None,
) -> list[DiscoveredModel]:
    """Scan the primary model dir, then any extra dirs, then (optionally) the HF cache.

    Earlier sources win on id collision: an explicitly placed model shadows a cached
    copy of the same name. The HF cache is opt-in so the catalogue lists only what the
    user placed in their model dirs unless they ask for the shared cache too.
    """
    found = discover_local(models_dir)
    seen = {m.id for m in found}

    def _add(models: list[DiscoveredModel]) -> None:
        for m in models:
            if m.id not in seen:
                found.append(m)
                seen.add(m.id)

    for d in extra_dirs or []:
        _add(discover_local(d))
    if include_hf_cache:
        _add(discover_hf_cache(hf_cache_dir))
    return found
