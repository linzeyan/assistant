"""Native text-to-video backend via Blaizzy's mlx-video (Wan / LTX on MLX).

Roadmap Phase 6: local video generation. Optional and defensive like the other MLX
backends — ``available()`` reports importability and generation runs in a worker
thread. Video is the heaviest MLX workload, so keeping it off the event loop matters
most here (plan Part E, risk 2).

The real package is Blaizzy's ``mlx-video`` (installed from git, NOT the identically
named PyPI I/O library): generation lives at
``mlx_video.models.{wan_2,ltx_2}.generate.generate_video`` and loads weights from a
local *converted-MLX* checkpoint dir passed as ``model_dir`` (e.g. ``Wan2.2-TI2V-5B-mlx``).
The pipeline is chosen by name: ``"wan"`` (``wan_2``) or ``"ltx"`` (``ltx_2``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


def _valid_num_frames(n: int) -> int:
    """Coerce a frame count to the Wan/LTX-required ``4·k + 1`` (the pipeline asserts this).

    The tool exposes a free integer and local models pick arbitrary values, so round down to
    the nearest valid count (min 5) instead of letting generation hard-fail on a bad request.
    """
    n = max(5, n)
    return n - ((n - 1) % 4)


class VideoService(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def generate_video(
        self, prompt: str, *, num_frames: int | None = None, seed: int | None = None
    ) -> Path:
        """Generate a short video and return the path to the saved file."""


class MlxVideoBackend(VideoService):
    def __init__(
        self, video_dir: Path, model: str = "wan", *, checkpoint: Path | None = None
    ):
        self._dir = Path(video_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        # Local converted-MLX checkpoint dir handed to mlx-video as ``model_dir``. Unlike
        # image gen (mflux resolves an alias internally), Wan/LTX need an explicit on-disk
        # checkpoint — without it generation can't run. Runtime-switchable via set_checkpoint
        # so the Telegram /video picker (N28) can change it without a restart.
        self._checkpoint = Path(checkpoint) if checkpoint else None

    @property
    def checkpoint(self) -> Path | None:
        return self._checkpoint

    def set_checkpoint(self, checkpoint: Path | None) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else None

    def available(self) -> bool:
        # Check the REAL generation submodule, not just top-level ``mlx_video``: the unrelated
        # PyPI "mlx-video" I/O lib also imports as ``mlx_video`` but has no ``models.wan_2``,
        # so a top-level check falsely reports availability against the wrong package.
        try:
            return (
                importlib.util.find_spec("mlx_video.models.wan_2.generate") is not None
            )
        except ImportError:
            return False

    async def generate_video(
        self, prompt: str, *, num_frames: int | None = None, seed: int | None = None
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "video requires Blaizzy's mlx-video. "
                'Install with: uv pip install -e ".[video]"'
            )
        return await asyncio.to_thread(self._generate_sync, prompt, num_frames, seed)

    def _generate_sync(
        self, prompt: str, num_frames: int | None, seed: int | None
    ) -> Path:
        if self._checkpoint is None or not self._checkpoint.is_dir():
            raise RuntimeError(
                "video generation needs a local converted-MLX checkpoint; set "
                "`video_checkpoint` to a Wan/LTX model dir (e.g. .../Wan2.2-TI2V-5B-mlx)."
            )
        # Single integration point with mlx-video: "wan" -> models.wan_2, "ltx" -> models.ltx_2.
        # Both expose generate_video(model_dir, prompt, ..., output_path); adjusting to a new
        # package version touches only here.
        if self._model == "ltx":
            from mlx_video.models.ltx_2.generate import generate_video as _generate
        else:
            from mlx_video.models.wan_2.generate import generate_video as _generate

        out = self._dir / f"vid_{uuid.uuid4().hex[:8]}.mp4"
        kwargs: dict = {}
        if num_frames is not None:
            kwargs["num_frames"] = _valid_num_frames(num_frames)
        if seed is not None:
            kwargs["seed"] = seed
        # generate_video writes to output_path and returns None — return the path we chose.
        _generate(
            model_dir=str(self._checkpoint),
            prompt=prompt,
            output_path=str(out),
            **kwargs,
        )
        return out
