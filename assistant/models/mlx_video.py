"""Native text-to-video backend via mlx-video (Wan / LTX on MLX).

Roadmap Phase 6: local video generation. Optional and defensive like the other MLX
backends — ``available()`` reports importability and generation runs in a worker
thread. Video is the heaviest MLX workload, so keeping it off the event loop matters
most here (plan Part E, risk 2). The pipeline is chosen by name: ``"wan"``
(``mlx_video.wan_2``) or ``"ltx"`` (``mlx_video.ltx_2``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


class VideoService(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def generate_video(
        self, prompt: str, *, num_frames: int | None = None, seed: int | None = None
    ) -> Path:
        """Generate a short video and return the path to the saved file."""


class MlxVideoBackend(VideoService):
    def __init__(self, video_dir: Path, model: str = "wan"):
        self._dir = Path(video_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model

    def available(self) -> bool:
        return importlib.util.find_spec("mlx_video") is not None

    async def generate_video(
        self, prompt: str, *, num_frames: int | None = None, seed: int | None = None
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                'video requires mlx-video. Install with: uv pip install -e ".[video]"'
            )
        return await asyncio.to_thread(self._generate_sync, prompt, num_frames, seed)

    def _generate_sync(
        self, prompt: str, num_frames: int | None, seed: int | None
    ) -> Path:
        # Single integration point with mlx-video: "wan" -> mlx_video.wan_2,
        # "ltx" -> mlx_video.ltx_2. Adjusting to the installed version touches only here.
        if self._model == "ltx":
            from mlx_video.ltx_2 import generate as pipeline
        else:
            from mlx_video.wan_2 import generate as pipeline

        out = self._dir / f"vid_{uuid.uuid4().hex[:8]}.mp4"
        pipeline.generate(
            prompt=prompt,
            output_path=str(out),
            num_frames=num_frames or 73,
            seed=seed if seed is not None else 0,
        )
        return out
