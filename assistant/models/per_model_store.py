"""Per-model overrides (oMLX-style), keyed by model id and persisted to disk.

Three independent concerns share one store/file/API:

- **Generation** — sampler settings (temperature / top_p / top_k) and an optional max_tokens cap,
  applied at chat time layered over the global defaults, so a creative model and a deterministic
  coder can coexist without a config edit.
- **Type override** — force a model's kind (llm / vlm / image / video / embed) instead of trusting
  auto-detection. This is how a checkpoint we misclassify (e.g. OptiQ quants that carry the omni
  base's vision_config but only load through mlx-lm) is told to load as a plain ``llm`` and just
  work. "auto" clears the override and returns to detection.
- **Chat-template kwargs** — variables forwarded into the model's chat-template jinja context on
  every render (e.g. Qwen3.x ``enable_thinking: false``, which swaps the open ``<think>`` in the
  generation prompt for an empty block so the model answers directly). Templates ignore variables
  they don't know, so these are safe to store for any model.
- **Draft pairing** — the model id of a speculative drafter (an MTP/DFlash/EAGLE checkpoint,
  kind "draft") to decode this model with. Set → the model loads through mlx-vlm's speculative
  path; unset → plain decoding.

The concerns are kept apart at read time: generation params are merged into sampling kwargs, but
``type``/``chat_template_kwargs`` must NOT be (a stray kwarg would break generation, and the
template kwargs need dict-merge semantics), so each has its own getter.
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
VALID_TYPES = ("llm", "vlm", "image", "video", "embedding", "audio", "draft")


def _clean_template_kwargs(value) -> dict:
    """A usable chat_template_kwargs value: a dict of str -> JSON scalar. Non-scalar values are
    dropped (a template variable is a jinja scalar; nesting is never meaningful there) and a
    non-dict clears the whole entry."""
    if not isinstance(value, dict):
        return {}
    return {
        k: v
        for k, v in value.items()
        if isinstance(k, str) and isinstance(v, (bool, int, float, str))
    }


def _clean(settings: dict) -> dict:
    """Keep only known keys with usable values (a null/absent value clears that override).

    Generation keys keep non-null values; ``type`` is kept only when it's a recognised kind (an
    "auto"/unknown value clears it, so the model falls back to auto-detection)."""
    out = {k: settings[k] for k in ALLOWED_KEYS if settings.get(k) is not None}
    t = settings.get("type")
    if isinstance(t, str) and t.strip().lower() in VALID_TYPES:
        out["type"] = t.strip().lower()
    # Speculative-decoding drafter: the model id of an MTP/DFlash/EAGLE checkpoint to pair with
    # this model at load time. Just a reference — resolution/compatibility is checked at load,
    # where the drafter's weights and the target actually meet.
    d = settings.get("draft")
    if isinstance(d, str) and d.strip():
        out["draft"] = d.strip()
    tpl = _clean_template_kwargs(settings.get("chat_template_kwargs"))
    if tpl:
        out["chat_template_kwargs"] = tpl
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

    def draft_model(self, model_id: str) -> str | None:
        """The paired speculative drafter's model id, or None when the model decodes plainly."""
        d = self._data.get(model_id, {}).get("draft")
        return d if isinstance(d, str) and d else None

    def chat_template_kwargs(self, model_id: str) -> dict:
        """This model's saved chat-template variables ({} when none). Merged into the
        ``chat_template_kwargs`` stream param at chat time — dict-merge, stored keys win —
        never into sampler kwargs."""
        return dict(self._data.get(model_id, {}).get("chat_template_kwargs", {}))

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
