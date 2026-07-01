"""Per-model overrides (oMLX-style), keyed by model id and persisted to disk.

Two independent concerns share one store/file/API:

- **Generation** — sampler settings (temperature / top_p / top_k) and an optional max_tokens cap,
  applied at chat time layered over the global defaults, so a creative model and a deterministic
  coder can coexist without a config edit.
- **Type override** — force a model's kind (llm / vlm / image / video / embed) instead of trusting
  auto-detection. This is how a checkpoint we misclassify (e.g. gemma-4-31b auto-detected as a VLM,
  which then crashes the mlx-vlm loader) is told to load as a plain ``llm`` and just work. "auto"
  clears the override and returns to detection.

The two are kept apart at read time: generation params are merged into sampling kwargs, but the
type override must NOT be (a stray ``type`` kwarg would break generation), so it has its own getter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("assistant")

# Sampler/generation keys — the only ones merged into generation kwargs.
ALLOWED_KEYS = ("temperature", "top_p", "top_k", "max_tokens")
# Kinds a user may force via the type override. "auto" is the UI's way to clear it (→ detection);
# it is never stored. These MUST match mlx_discovery.classify_kind's outputs so a forced kind
# routes the loader and filters into the right Models tab identically to a detected one.
VALID_TYPES = ("llm", "vlm", "image", "video", "embedding")


def _clean(settings: dict) -> dict:
    """Keep only known keys with usable values (a null/absent value clears that override).

    Generation keys keep non-null values; ``type`` is kept only when it's a recognised kind (an
    "auto"/unknown value clears it, so the model falls back to auto-detection)."""
    out = {k: settings[k] for k in ALLOWED_KEYS if settings.get(k) is not None}
    t = settings.get("type")
    if isinstance(t, str) and t.strip().lower() in VALID_TYPES:
        out["type"] = t.strip().lower()
    return out


class PerModelStore:
    def __init__(self, path: Path):
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                if isinstance(raw, dict):
                    self._data = {k: _clean(v) for k, v in raw.items() if isinstance(v, dict)}
            except (OSError, ValueError):
                log.warning("could not read per-model settings at %s", self._path)

    def get(self, model_id: str) -> dict:
        """The full public view (generation params + any type override) — for the settings API."""
        return dict(self._data.get(model_id, {}))

    def generation(self, model_id: str) -> dict:
        """Only the sampler/generation params — safe to merge into generation kwargs. Excludes the
        type override, which is metadata about loading, not a generation argument."""
        return {k: v for k, v in self._data.get(model_id, {}).items() if k in ALLOWED_KEYS}

    def kind_override(self, model_id: str) -> str | None:
        """The forced kind for this model, or None to auto-detect."""
        t = self._data.get(model_id, {}).get("type")
        return t if t in VALID_TYPES else None

    def set(self, model_id: str, settings: dict) -> dict:
        cleaned = _clean(settings)
        if cleaned:
            self._data[model_id] = cleaned
        else:
            self._data.pop(model_id, None)  # an all-null update clears the model's overrides
        self._persist()
        return self.get(model_id)

    def _persist(self) -> None:
        try:
            self._path.write_text(json.dumps(self._data))
        except OSError:
            log.exception("could not persist per-model settings to %s", self._path)
