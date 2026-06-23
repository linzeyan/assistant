"""Native text-embedding backend via mlx-embeddings.

Used to upgrade long-term memory from keyword overlap to semantic similarity. Like
the other MLX backends it is optional and defensive: ``available()`` reports whether
``mlx-embeddings`` is importable, and embedding runs in a worker thread (MLX is
blocking). The ``Embedder`` ABC is the seam the memory provider depends on, so tests
inject a deterministic fake instead of loading a real model.
"""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per input text."""


class MlxEmbeddingBackend(Embedder):
    def __init__(self, model: str = "mlx-community/bge-small-en-v1.5-bf16"):
        self._model = model
        self._cached: tuple | None = None  # (model, tokenizer) loaded once, reused

    def available(self) -> bool:
        return importlib.util.find_spec("mlx_embeddings") is not None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.available():
            raise RuntimeError(
                "embeddings require mlx-embeddings. Install with: "
                'uv pip install -e ".[embeddings]"'
            )
        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        # Single integration point with mlx-embeddings: adjusting to an installed
        # version touches only this method.
        from mlx_embeddings import generate, load

        if self._cached is None:
            self._cached = load(self._model)
        model, tokenizer = self._cached
        output = generate(model, tokenizer, texts)
        # ``text_embeds`` are L2-normalised pooled vectors; tolist() -> plain floats.
        return output.text_embeds.tolist()
