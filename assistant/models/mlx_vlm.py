"""Native vision-language backend via mlx-vlm.

Lets the assistant "read" images: given an image path and a question, a VLM returns
a text answer. Exposed to the agent as the ``view_image`` tool, so the main (text)
LLM can call it mid-conversation — no need for the chat model itself to be
multimodal. Optional and defensive like the other MLX backends; the model is loaded
once and cached (VLMs are large — see the unified-memory risk in plan Part E).
"""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC, abstractmethod


class VisionService(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def describe(self, image_paths: list[str], prompt: str) -> str:
        """Answer ``prompt`` about the given image(s) and return the text."""


class MlxVLMBackend(VisionService):
    def __init__(self, model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"):
        self._model = model
        self._cached: tuple | None = None  # (model, processor, config) loaded once

    def available(self) -> bool:
        return importlib.util.find_spec("mlx_vlm") is not None

    async def describe(self, image_paths: list[str], prompt: str) -> str:
        if not self.available():
            raise RuntimeError(
                'vision requires mlx-vlm. Install with: uv pip install -e ".[vlm]"'
            )
        return await asyncio.to_thread(self._describe_sync, image_paths, prompt)

    def _describe_sync(self, image_paths: list[str], prompt: str) -> str:
        # Single integration point with mlx-vlm: adjusting to an installed version
        # touches only this method.
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        if self._cached is None:
            model, processor = load(self._model)
            self._cached = (model, processor, load_config(self._model))
        model, processor, config = self._cached
        formatted = apply_chat_template(
            processor, config, prompt, num_images=len(image_paths)
        )
        output = generate(model, processor, formatted, image_paths, verbose=False)
        return output if isinstance(output, str) else str(output)
