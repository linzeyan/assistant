"""The model-layer seam (plan Part A).

``ModelService`` isolates the agent + API layers from the model backend. Today the
only implementation is ``OmlxModelService`` (A1: omlx-over-HTTP). A future in-process
MLX implementation (A3) can replace it without touching anything upstream.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from .omlx_client import OmlxClient
from .omlx_subprocess import OmlxProcess, OmlxStatus
from .types import ModelInfo


class ModelService(ABC):
    @abstractmethod
    async def start(self) -> OmlxStatus: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def reachable(self) -> bool: ...

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def load(self, model_id: str) -> None: ...

    @abstractmethod
    async def unload(self, model_id: str) -> None: ...

    @abstractmethod
    def stream_chat(
        self, messages: list[dict], model: str, tools: list[dict] | None = None, **params
    ) -> AsyncIterator[dict]: ...


class OmlxModelService(ModelService):
    def __init__(self, client: OmlxClient, process: OmlxProcess):
        self._client = client
        self._process = process
        self._status: OmlxStatus | None = None

    async def start(self) -> OmlxStatus:
        self._status = await self._process.ensure_running()
        return self._status

    async def stop(self) -> None:
        await self._process.stop()

    @property
    def status(self) -> OmlxStatus | None:
        return self._status

    async def reachable(self) -> bool:
        return await self._client.health()

    async def list_models(self) -> list[ModelInfo]:
        # Fail soft: an unreachable backend yields an empty list, never an exception,
        # so the GUI can still render and show the omlx status banner.
        if not await self._client.health():
            return []
        return await self._client.list_models()

    async def load(self, model_id: str) -> None:
        await self._client.load(model_id)

    async def unload(self, model_id: str) -> None:
        await self._client.unload(model_id)

    def stream_chat(
        self, messages: list[dict], model: str, tools: list[dict] | None = None, **params
    ) -> AsyncIterator[dict]:
        return self._client.stream_chat(messages, model, tools=tools, **params)
