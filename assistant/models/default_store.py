"""Backend-authoritative default chat model, shared by the API and the Telegram gateway.

The Mac app's "Default" button used to persist only to client-side UserDefaults, so the
Telegram gateway (which read the config's default_model) could never match it. This makes the
default a single backend-persisted value: the GUI writes it via PUT /models/default and both
the GUI and Telegram read the same source. Seeded from config.default_model on first run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("assistant")


class DefaultModelStore:
    def __init__(self, path: Path, seed: str | None = None):
        self._path = Path(path)
        self._value = seed
        if self._path.exists():
            try:
                stored = json.loads(self._path.read_text()).get("default")
                if stored:
                    self._value = stored
            except (OSError, ValueError):
                log.warning("could not read default-model store at %s", self._path)

    @property
    def value(self) -> str | None:
        return self._value

    def set(self, model_id: str | None) -> None:
        self._value = model_id or None
        try:
            self._path.write_text(json.dumps({"default": self._value}))
        except OSError:
            log.exception("could not persist default model to %s", self._path)
