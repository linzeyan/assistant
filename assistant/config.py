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

    # --- native modality models (mlx-* backends; used when the extra is installed) ---
    embed_memory: bool = True  # embeddings-backed semantic memory search
    embed_model: str = "mlx-community/bge-small-en-v1.5-bf16"
    vlm_model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    stt_model: str = "mlx-community/whisper-tiny"
    tts_model: str = "mlx-community/Kokoro-82M-bf16"

    # --- model lifecycle / memory guardrails (MLX unified memory is the top risk) ---
    max_loaded_models: int = 1
    mem_ceiling_gb: float | None = None

    # --- paths (XDG data dir) ---
    home_dir: Path = XDG_DATA_DIR
    skills_dir: Path = XDG_DATA_DIR / "skills"
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
    # Where over-budget tool output (S4) is spilled in full so the agent can read the rest.
    tool_output_dir: Path = XDG_DATA_DIR / "tool-output"
    # Directory the coding/shell tools operate in. Defaults to the process cwd.
    workspace_dir: Path = Field(default_factory=Path.cwd)
    # Fallback model for non-GUI entry points (Telegram) that don't pick one.
    default_model: str | None = None

    # --- image generation / editing ---
    image_model: str = "schnell"  # mflux alias (schnell/dev); used when mflux present
    images_dir: Path = XDG_DATA_DIR / "images"
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
