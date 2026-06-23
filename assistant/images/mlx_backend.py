"""MLX image generation + editing backend via mflux (FLUX / Qwen-Image-Edit on Apple
Silicon).

mflux is an optional, heavy, Apple-Silicon-only dependency, so this backend is
defensive: ``available()`` reports whether mflux is importable, and inference is
offloaded to a worker thread because MLX is synchronous and blocking — running it on
the event loop would stall chat streaming and the Telegram gateway (plan Part E, risk 2).

Targets mflux >= 0.18 (text-to-image via Flux1, image editing via QwenImageEdit). The
mflux API has moved across versions (top-level ``Config``/``Flux1`` were removed and
``generate_image`` now takes flat kwargs instead of a ``Config`` object); all imports are
isolated in the sync methods so a version bump touches only those.
"""

from __future__ import annotations

import asyncio
import importlib.util
import uuid
from pathlib import Path

from .service import MediaService


class MlxImageBackend(MediaService):
    def __init__(
        self, images_dir: Path, model: str = "schnell", *, edit_quantize: int | None = None
    ):
        self._dir = Path(images_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        # Optional quantization for the large Qwen-Image-Edit model (8/4) to fit tighter
        # unified memory; None = full precision.
        self._edit_quantize = edit_quantize

    def available(self) -> bool:
        return importlib.util.find_spec("mflux") is not None

    async def generate_image(
        self,
        prompt: str,
        *,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "image generation requires mflux. Install it with: pip install mflux"
            )
        return await asyncio.to_thread(
            self._generate_sync, prompt, steps, seed, width, height
        )

    def _generate_sync(
        self,
        prompt: str,
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
    ) -> Path:
        # Lazy, isolated import (mflux >= 0.18 layout): Flux1 lives under the txt2img
        # variant package and generate_image takes flat kwargs (no Config object).
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name(self._model)
        image = flux.generate_image(
            seed=seed if seed is not None else 0,
            prompt=prompt,
            num_inference_steps=steps or (4 if self._model == "schnell" else 20),
            height=height or 1024,
            width=width or 1024,
        )
        out = self._dir / f"img_{uuid.uuid4().hex[:8]}.png"
        image.save(path=str(out))
        return out

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
        if not self.available():
            raise RuntimeError(
                "image editing requires mflux>=0.18 (uv pip install -U mflux)"
            )
        return await asyncio.to_thread(
            self._edit_sync, prompt, list(image_paths), steps, seed, width, height, guidance
        )

    def _edit_sync(
        self,
        prompt: str,
        image_paths: list[str],
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
        guidance: float | None,
    ) -> Path:
        # Qwen-Image-Edit via mflux >= 0.18. The qwen edit module is absent on older mflux,
        # so an ImportError here is surfaced as a clear "upgrade mflux" rather than a crash.
        try:
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        except ImportError as exc:
            raise RuntimeError(
                "this mflux build has no Qwen image edit; upgrade to mflux>=0.18 "
                "(Settings > Managed tools > 更新套件) and restart the backend."
            ) from exc

        editor = QwenImageEdit(quantize=self._edit_quantize)
        # Default to 8 steps / guidance 2.5 (mflux's default of 4 is Lightning-LoRA tuned
        # and looks broken for a normal edit); height/width default to the input's size.
        kwargs: dict = {
            "seed": seed if seed is not None else 0,
            "prompt": prompt,
            "image_paths": image_paths,
            "num_inference_steps": steps or 8,
            "guidance": guidance if guidance is not None else 2.5,
        }
        if height:
            kwargs["height"] = height
        if width:
            kwargs["width"] = width
        image = editor.generate_image(**kwargs)
        out = self._dir / f"edit_{uuid.uuid4().hex[:8]}.png"
        image.save(path=str(out))
        return out
