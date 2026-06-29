"""Per-model generation overrides (oMLX-style), keyed by model id and persisted to disk.

Each model can carry its own sampler settings (temperature / top_p / top_k) and an optional
max_tokens cap. They're applied at chat time by the model service, layered over the global
defaults, so a creative model and a deterministic coder can coexist without a config edit.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("assistant")

# The only keys we persist/apply. Anything else in a PUT body is ignored, so the store can't be
# used to smuggle arbitrary kwargs into generation.
ALLOWED_KEYS = ("temperature", "top_p", "top_k", "max_tokens")


def _clean(settings: dict) -> dict:
    """Keep only known keys with non-null values (a null clears that override)."""
    return {k: settings[k] for k in ALLOWED_KEYS if settings.get(k) is not None}


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
        return dict(self._data.get(model_id, {}))

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
