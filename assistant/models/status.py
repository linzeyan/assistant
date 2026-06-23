"""Backend-agnostic status types for the model layer.

Originally these lived in ``omlx_subprocess`` as ``OmlxState``/``OmlxStatus``. Now
that the project has two model backends (external omlx and the native in-process MLX
backend), the status carrier is shared and named generically. ``omlx_subprocess``
re-exports the old names for back-compat.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BackendState(str, Enum):
    CONNECTED = "connected"  # reused an already-running external server (we did not spawn)
    SPAWNED = "spawned"  # we started an external server and own its lifecycle
    LOCAL = "local"  # in-process backend ready (no external server)
    UNAVAILABLE = "unavailable"  # not installed / could not become healthy


@dataclass
class BackendStatus:
    state: BackendState
    detail: str
    base_url: str
