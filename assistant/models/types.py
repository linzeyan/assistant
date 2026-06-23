from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """A model omlx knows about.

    ``loaded`` reflects whether the model currently occupies memory in omlx's
    EnginePool, as opposed to merely being discovered on disk. We keep this struct
    minimal — the GUI needs id/type/loaded/source and nothing more for switching.
    """

    id: str
    type: str | None = None
    loaded: bool = False
    source: str | None = None
    size_bytes: int = 0  # on-disk footprint, for display + capacity management
