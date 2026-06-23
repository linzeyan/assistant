from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class MediaService(ABC):
    """Seam for local media generation.

    Phase 5 ships image generation; the same interface leaves room for a
    ``generate_video`` method later (plan Phase 6) behind one backend abstraction.
    """

    @abstractmethod
    def available(self) -> bool:
        """Whether the backend can actually generate (deps installed, etc.)."""

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        *,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        """Generate an image and return the path to the saved file."""

    @abstractmethod
    async def edit_image(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
    ) -> Path:
        """Edit/transform the given input image(s) per the prompt (image-to-image, e.g.
        Qwen-Image-Edit) and return the path to the saved file."""
