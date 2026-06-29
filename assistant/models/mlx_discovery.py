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
    # diffusers image/video pipelines (FLUX, Qwen-Image) ship model_index.json instead of a
    # top-level config.json; accepting it keeps those checkpoints from silently vanishing from
    # the Models list (they still need real weights via _has_weights).
    return (d / "config.json").is_file() or (d / "model_index.json").is_file()


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
# Diffusion image-pipeline class/arch markers (FLUX, Qwen-Image via mflux). A diffusers
# pipeline declares its class in model_index.json (_class_name "FluxPipeline" / "QwenImage…").
_IMAGE_CLASS_MARKERS = ("flux", "qwenimage", "qwen_image", "stablediffusion")


def _read_config(d: Path) -> dict:
    try:
        return json.loads((d / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def _read_model_index(d: Path) -> dict:
    """diffusers pipelines (FLUX / Qwen-Image) declare their class in model_index.json, not
    config.json — read it so classify_kind can tell an image pipeline from a chat model."""
    try:
        return json.loads((d / "model_index.json").read_text())
    except (OSError, ValueError):
        return {}


def _is_image_model(cfg: dict, d: Path) -> bool:
    archs = cfg.get("architectures") or []
    blob = f"{(archs[0] if archs else '')} {cfg.get('_class_name') or ''}".lower()
    if any(m in blob for m in _IMAGE_CLASS_MARKERS) and "video" not in blob:
        return True
    cls = str(_read_model_index(d).get("_class_name") or "").lower()
    return any(m in cls for m in _IMAGE_CLASS_MARKERS) and "video" not in cls


def classify_kind(d: Path) -> str:
    cfg = _read_config(d)
    archs = cfg.get("architectures") or []
    arch = (archs[0] if archs else "").lower()
    mtype = str(cfg.get("model_type") or "").lower()
    cls_name = str(cfg.get("_class_name") or "").lower()
    # A diffusers pipeline declares its class in model_index.json (no config.json); fold it in
    # so a pipeline-only Wan/LTX dir still routes to video instead of fail-opening to "llm".
    idx_cls = str(_read_model_index(d).get("_class_name") or "").lower()
    # Diffusion video models (Wan S2V/TI2V/T2V/I2V …) declare a video model_type or a
    # WanModel diffusers class. They are NOT chattable: without this they fail-open to
    # "llm" below and pollute the chat model list, where loading one as an LLM dies with
    # "No safetensors found" (weights are diffusion_pytorch_model-*, not LM format).
    if mtype in _VIDEO_TYPES or any(
        "wan" in c or "video" in c or "ltx" in c for c in (cls_name, idx_cls)
    ):
        return "video"
    # Diffusion image pipelines (FLUX / Qwen-Image via mflux). Checked before the vae→video
    # rule below because an image checkpoint also ships a VAE — the FLUX/Qwen-Image pipeline
    # class is the distinguishing signal. Keeps them under the Models "Image" tab rather than
    # fail-opening to "llm".
    if _is_image_model(cfg, d):
        return "image"
    # A chat-arch config (Qwen-VL / causal-LM) that ALSO ships a diffusion VAE is really a
    # generation / "omni" checkpoint, not a clean chat model — e.g. Lance-3B-Video declares
    # model_type qwen2_5_vl but carries extra generation heads + a vae.safetensors that
    # mlx-vlm's stock class can't load; picked as a chat model it dumped 1606 mismatched
    # weights (N32). A normal mlx chat VLM never bundles a diffusion VAE, so this file is a
    # safe signal to keep such models OUT of the chat picker (only mlx-loadable chat models
    # belong there). It isn't a loadable mlx-video checkpoint either (no t5_encoder), so the
    # /video picker's is_video_checkpoint filter excludes it too.
    if (d / "vae.safetensors").is_file():
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


def is_video_checkpoint(d: Path) -> bool:
    """True if ``d`` is a *converted-MLX* Wan/LTX checkpoint that mlx-video can actually load.

    A "video"-kind dir is not enough: the raw HuggingFace download (``Wan2.2-TI2V-5B`` with
    ``diffusion_pytorch_model-*.safetensors`` + ``Wan2.x_VAE.pth``) classifies as video but
    mlx-video CANNOT load it — only the converted layout can. That layout always carries a
    ``vae.safetensors`` + ``t5_encoder.safetensors`` plus at least one transformer weight
    (``model.safetensors`` for single-model, ``low_noise_model.safetensors`` for dual). This
    structural check keeps unloadable raw dirs out of the generation picker (N28).
    """
    d = Path(d)
    if not (d / "vae.safetensors").is_file() or not (d / "t5_encoder.safetensors").is_file():
        return False
    return (d / "model.safetensors").is_file() or (d / "low_noise_model.safetensors").is_file()


def discover_video_checkpoints(dirs: list[Path]) -> list[DiscoveredModel]:
    """Loadable mlx-video checkpoints across the given model dirs (flat + org/<model>
    layouts, via discover_local), deduped by id with earlier dirs winning. Only converted-MLX
    dirs qualify (see is_video_checkpoint), so the Telegram /video picker never offers a raw
    HF Wan dir that would fail at generation time."""
    found: list[DiscoveredModel] = []
    seen: set[str] = set()
    for base in dirs:
        for m in discover_local(base):
            if m.kind == "video" and m.id not in seen and is_video_checkpoint(m.path):
                found.append(m)
                seen.add(m.id)
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
