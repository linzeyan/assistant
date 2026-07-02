"""Single, small configuration surface.

Deliberately flat and ~17 fields — a direct reaction to hermes-agent's 20+ nested
config sections, which were a primary "not nice to use" pain point. Values come
from (in priority order) constructor args, ``ASSISTANT_*`` env vars, then an
optional ``~/.assistant/config.toml``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)


def _xdg_dir(env_var: str, default_rel: str) -> Path:
    # XDG Base Directory spec: honour the env var when set, else the spec default.
    raw = os.environ.get(env_var)
    base = Path(raw) if raw else Path.home() / default_rel
    return base / "assistant"


# Config under $XDG_CONFIG_HOME, all app data (skills/memory/images/audio/models)
# under $XDG_DATA_HOME. The HuggingFace cache stays at its own XDG cache location.
XDG_CONFIG_DIR = _xdg_dir("XDG_CONFIG_HOME", ".config")
XDG_DATA_DIR = _xdg_dir("XDG_DATA_HOME", ".local/share")


class Settings(BaseSettings):
    # protected_namespaces=(): we use plain field names like `models_dir`; silence
    # pydantic's `model_`-prefix guard which is irrelevant here.
    model_config = SettingsConfigDict(
        env_prefix="ASSISTANT_", extra="ignore", protected_namespaces=()
    )

    # --- backend: what the SwiftUI app / Telegram talk to ---
    backend_host: str = "127.0.0.1"  # set 0.0.0.0 to expose on the LAN
    backend_port: int = 9981

    # --- model backend selection ---
    # "mlx": native in-process MLX (mlx-lm), no external server — the default, so the
    #        app needs no omlx install. "omlx": external omlx server (connect-or-spawn).
    model_backend: str = "mlx"
    # Off by default: only models under the configured model dir(s) are listed, so the
    # catalogue reflects what the user actually placed there. Opt in to also surface the
    # shared HuggingFace hub cache (incidental downloads like the embedding model).
    hf_cache: bool = False

    # --- omlx model backend (A1: managed subprocess + OpenAI-compatible API) ---
    omlx_host: str = "127.0.0.1"
    omlx_port: int = 8000
    omlx_api_key: str | None = None
    omlx_bin: str | None = None  # explicit path; otherwise auto-detected (Homebrew)
    omlx_autostart: bool = True  # spawn `omlx serve` if not already running
    models_dir: Path = XDG_DATA_DIR / "models"
    # Additional directories to scan for models, beyond the primary models_dir. Lets the
    # user point at pre-existing model collections without moving them.
    extra_model_dirs: list[Path] = Field(default_factory=list)
    # Where /models/download fetches into. Defaults to models_dir so a downloaded
    # model is immediately discoverable in /models (no separate import step).
    download_dir: Path = XDG_DATA_DIR / "models"
    # Disable HuggingFace's Xet transfer for downloads (sets HF_HUB_DISABLE_XET=1). On by default:
    # measured throttling to a few KB/s on some networks, so leaving Xet on can cripple a download.
    # Set false to use Xet where it performs well.
    hf_hub_disable_xet: bool = True
    # Per-file connect/read timeout (seconds) for hub downloads (HF_HUB_DOWNLOAD_TIMEOUT). HF's own
    # default (~10s) is too tight on a slow link and aborts large shards mid-transfer.
    hf_hub_download_timeout: int = 120
    # Concurrent file downloads for snapshot_download(max_workers=...). Each file is a single HTTP
    # connection (Xet off), so ONE worker caps at that connection's throughput (~2 MB/s observed);
    # parallel files are what reach a link's full bandwidth. Default 8 to match the `huggingface-cli`
    # default (the speed users compare against); lower it only for a rate-limited/flaky connection.
    hf_download_max_workers: int = 8

    # --- native modality models (mlx-* backends; used when the extra is installed) ---
    embed_memory: bool = True  # embeddings-backed semantic memory search
    embed_model: str = "mlx-community/bge-small-en-v1.5-bf16"
    vlm_model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    stt_model: str = "mlx-community/whisper-tiny"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"

    # --- model lifecycle / memory guardrails (MLX unified memory is the top risk) ---
    # 0 = no count cap: models stay resident while they fit under the memory ceiling (which
    # defaults to physical RAM × 0.9), so switching/fusion doesn't thrash reloads. Set a positive
    # number to restore strict count-based eviction; the pool falls back to 1 if no ceiling can
    # be determined.
    max_loaded_models: int = 0
    mem_ceiling_gb: float | None = None

    # --- paths (XDG data dir) ---
    home_dir: Path = XDG_DATA_DIR
    skills_dir: Path = XDG_DATA_DIR / "skills"
    # Optional third skills source (E1): a project/team skills dir scanned LAST, so its skills win
    # on a slug collision (shadowing user + bundled). None = only bundled + user dirs. A fixed
    # configured path, not per-workspace — that would make the skills index (part of the stable
    # cacheable system prefix, S2/S3) vary per turn.
    project_skills_dir: Path | None = None
    memory_dir: Path = XDG_DATA_DIR / "memory"
    sessions_dir: Path = XDG_DATA_DIR / "sessions"  # persisted conversations (S1)
    # Per-turn trace (spring2 P0): record each turn (model text + parsed calls + tool
    # results) so "local-model turns don't always succeed" becomes a scannable, debuggable
    # list instead of a vibe. Records only — fixes nothing (measure-before-fix). Toggle off
    # to skip the per-turn disk writes.
    trace_enabled: bool = True
    trace_dir: Path = XDG_DATA_DIR / "traces"
    audio_dir: Path = XDG_DATA_DIR / "audio"

    # --- logging ---
    # The GUI spawns the backend with no console attached, so without an explicit file sink
    # its logs vanish — exactly when you need them to diagnose a failing turn. The server
    # entry point writes a rotating file here; ASSISTANT_LOG_LEVEL tunes verbosity.
    log_dir: Path = XDG_DATA_DIR / "logs"
    log_level: str = "INFO"

    # --- managed-tool install sources ---
    # Per-feature pip install-source overrides (key = feature: mlx/images/embeddings/vlm/
    # audio/video). When set, install/update targets this spec instead of the PyPI package
    # — e.g. a patched mlx-lm build supporting newer model architectures than the published
    # wheel (qwen3_5/qwen3_6). Such tools update by re-pulling the source (forced reinstall),
    # never a PyPI --upgrade that would clobber the patched build. In config.toml:
    #   [managed_tool_sources]
    #   mlx = "git+https://github.com/ml-explore/mlx-lm.git@refs/pull/1192/head"
    managed_tool_sources: dict[str, str] = Field(default_factory=dict)

    # --- conversation compaction (S6) ---
    # When estimated context exceeds (window - reserve), the oldest turns are summarized so a
    # long session stays within the model's window. The window is auto-detected from the model
    # when possible; compaction_context_window is the fallback. keep_recent is kept verbatim.
    compaction_enabled: bool = True
    compaction_context_window: int = 8192
    compaction_reserve_tokens: int = 1024
    compaction_keep_recent_tokens: int = 3072

    # --- agent behaviour ---
    approval_required: bool = True
    # Wildcard permission rules (S5): each is {action = "<tool-name glob>", resource =
    # "<glob, default *>", decision = "allow|deny|ask"}. First match wins; no match falls
    # through to the normal prompt. Lets the user pre-authorise safe tools or block dangerous
    # ones without a prompt each time. In config.toml:
    #   [[approval_rules]]
    #   action = "read_file"
    #   decision = "allow"
    approval_rules: list[dict] = Field(default_factory=list)
    # Remember an interactively-granted (tool, resource) so the same action isn't re-prompted.
    # Process-scoped: resets when the backend restarts.
    approval_ask_once: bool = True
    # Cap on tokens generated per assistant turn. The MLX engine defaulted to 1024, which
    # silently truncated long answers (code, explanations) mid-output; this is the real
    # ceiling and is config-tunable. Generation still stops early on the model's EOS.
    max_output_tokens: int = 4096
    # Max tool-calling iterations per turn before the loop gives up. Spring4 SB.3 measured a
    # skill-driven turn (investigate: skill_view + reproduce + read + git log + git show + fix +
    # regression test) needing well past the old default of 8 — a multi-step task hit the ceiling
    # mid-investigation. 16 fits a full skill workflow with some retries while still bounding a
    # runaway loop (worst case ~16 generations/turn). Tunable for tighter/looser budgets.
    max_tool_iters: int = 16
    # Per-turn wall-clock budget in seconds (B1); None = unlimited (default, so a legitimately
    # slow large-model turn is never killed). When set, the loop aborts a runaway turn BETWEEN
    # iterations (a stuck tool-call loop) with a loud error. Note it can't preempt a single
    # in-flight generation (MLX has no token-level interrupt; that's bounded by max_output_tokens).
    turn_timeout_s: float | None = None
    # Where over-budget tool output (S4) is spilled in full so the agent can read the rest.
    tool_output_dir: Path = XDG_DATA_DIR / "tool-output"
    # Retention for spilled tool-output files (S15 GC), swept once at startup. 0 = keep forever.
    tool_output_retention_days: float = 14
    # Directory the coding/shell tools operate in. Defaults to the process cwd.
    workspace_dir: Path = Field(default_factory=Path.cwd)
    # Fallback model for non-GUI entry points (Telegram) that don't pick one.
    default_model: str | None = None

    # --- image generation / editing ---
    # Optional image-model preference, matched as a substring against discovered local image
    # checkpoints (resolved to the on-disk path at startup). Empty (default) → the smallest
    # discovered image checkpoint is used, so there's no hardcoded model name and the default is
    # the most memory-frugal one available. (Set to "schnell"/"dev" to use mflux's FLUX.1 directly
    # — note that pulls a multi-GB download on first use.)
    image_model: str = ""
    images_dir: Path = XDG_DATA_DIR / "images"
    # Default output size / steps, runtime-switchable from the Telegram /imageset picker and the
    # GUI. A per-request tool arg still overrides these. 512² by default — lighter/faster on
    # constrained memory; bump to 768/1024 per request or via /imageset. steps None lets the
    # backend pick a sane per-model default.
    image_default_width: int = 512
    image_default_height: int = 512
    image_default_steps: int | None = None
    # Optional quantization (8/4) for the large Qwen-Image-Edit model used by edit_image,
    # to fit tighter unified memory; None = full precision.
    image_edit_quantize: int | None = None

    # --- video generation ---
    video_model: str = "wan"  # mlx-video pipeline: "wan" (wan_2) or "ltx" (ltx_2)
    video_dir: Path = XDG_DATA_DIR / "videos"
    # Local converted-MLX checkpoint dir for video gen (mlx-video's `model_dir`). Unlike image
    # gen (mflux resolves an alias internally), Wan/LTX must be pointed at an on-disk checkpoint
    # — unset → generate_video returns a clear "set video_checkpoint" error. Picking this from
    # discovered models is the next sprint's job (N28); for now it's an explicit path, e.g.
    # .../Wan-AI/Wan2.2-TI2V-5B-mlx.
    video_checkpoint: Path | None = None

    # --- fusion (multi-model panel + judge, exposed as the virtual "fusion" model) ---
    # Seed values; the live config is persisted (home_dir/fusion.json) and editable at runtime
    # via PUT /fusion. Only takes effect with a non-empty panel AND a judge.
    fusion_enabled: bool = False
    fusion_panel: list[str] = Field(default_factory=list)
    fusion_judge: str | None = None

    # --- telegram (wired in a later phase) ---
    telegram_token: str | None = None
    telegram_allowed_users: list[int] = Field(default_factory=list)

    @property
    def omlx_base_url(self) -> str:
        return f"http://{self.omlx_host}:{self.omlx_port}"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        # Append TOML last (lowest priority) and only if present, so a missing file
        # is a no-op rather than an error.
        toml_path = XDG_CONFIG_DIR / "config.toml"
        if toml_path.is_file():
            sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
        return tuple(sources)


@lru_cache
def get_settings() -> Settings:
    return Settings()
