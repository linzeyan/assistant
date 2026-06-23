from __future__ import annotations

from collections.abc import AsyncIterator

from assistant.models.service import ModelService


class AsyncLLM:
    """Thin async wrapper over the model backend.

    Exposes the structured streaming turn (text deltas + assembled tool calls) that
    the agent loop consumes. Keeping it here means the loop depends on a small,
    stable interface rather than on the omlx client directly.
    """

    def __init__(self, models: ModelService):
        self._models = models

    def stream_chat(
        self, messages: list[dict], model: str, tools: list[dict] | None = None, **params
    ) -> AsyncIterator[dict]:
        return self._models.stream_chat(messages, model, tools=tools, **params)
