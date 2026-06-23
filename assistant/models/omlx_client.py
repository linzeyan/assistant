"""Async HTTP client for an omlx server's OpenAI-compatible + model-admin API.

We deliberately speak only omlx's stable HTTP surface (``/v1/...``) rather than
importing omlx in-process: omlx ships via Homebrew with a heavy, git-pinned MLX
dependency tree (mlx-lm/mlx-vlm/dflash + venvstacks packaging), so coupling to its
HTTP *contract* — not its code — insulates us from that churn. See plan Part A (A1).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from .types import ModelInfo


class OmlxClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 600.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        # Long default timeout: a single local generation stream can run for minutes.
        self._client = httpx.AsyncClient(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        """Cheap reachability probe used by the connect-or-spawn logic."""
        try:
            r = await self._client.get("/v1/models", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        r = await self._client.get("/v1/models")
        r.raise_for_status()
        data = r.json().get("data", [])
        loaded = await self._loaded_ids()
        out: list[ModelInfo] = []
        for m in data:
            mid = m.get("id")
            if not mid:
                continue
            out.append(
                ModelInfo(
                    id=mid,
                    type=m.get("type") or m.get("model_type"),
                    loaded=mid in loaded,
                    source=m.get("source") or m.get("source_type"),
                    size_bytes=m.get("size_bytes") or 0,
                )
            )
        return out

    async def _loaded_ids(self) -> set[str]:
        """Best-effort set of currently-loaded model ids.

        ``/v1/models/status`` shape is not part of the stable contract, so we parse
        defensively and fail soft to an empty set — a wrong `loaded` badge is
        cosmetic and must never break model listing.
        """
        try:
            r = await self._client.get("/v1/models/status")
            if r.status_code != 200:
                return set()
            payload = r.json()
        except httpx.HTTPError:
            return set()
        items = payload.get("models") if isinstance(payload, dict) else payload
        if isinstance(items, dict):
            items = list(items.values())
        ids: set[str] = set()
        for it in items or []:
            if isinstance(it, dict) and it.get("loaded") and it.get("id"):
                ids.add(it["id"])
        return ids

    async def load(self, model_id: str) -> None:
        r = await self._client.post(f"/v1/models/{model_id}/load")
        r.raise_for_status()

    async def unload(self, model_id: str) -> None:
        r = await self._client.post(f"/v1/models/{model_id}/unload")
        r.raise_for_status()

    async def stream_chat(
        self,
        messages: list[dict],
        model: str,
        tools: list[dict] | None = None,
        **params,
    ) -> AsyncIterator[dict]:
        """Stream a single assistant turn as typed events.

        Yields ``{"type": "text", "content": str}`` for content deltas. If the model
        emits tool calls, their streamed fragments are accumulated by index and a
        single ``{"type": "tool_calls", "tool_calls": [...]}`` event is yielded once
        the stream completes — assembling streamed tool-call deltas is the fiddly
        part the agent loop should not have to know about.
        """
        body: dict = {"model": model, "messages": messages, "stream": True, **params}
        if tools:
            body["tools"] = tools
        acc: dict[int, dict] = {}
        async with self._client.stream(
            "POST", "/v1/chat/completions", json=body
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                if delta.get("content"):
                    yield {"type": "text", "content": delta["content"]}
                for tc in delta.get("tool_calls") or []:
                    slot = acc.setdefault(
                        tc.get("index", 0), {"id": None, "name": None, "args": ""}
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
        if acc:
            assembled = []
            for index in sorted(acc):
                slot = acc[index]
                raw = slot["args"]
                try:
                    arguments = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    # Keep the raw fragment so the tool can report a clear error
                    # rather than the loop silently losing the call.
                    arguments = {"__raw_arguments__": raw}
                assembled.append(
                    {
                        "id": slot["id"] or f"call_{index}",
                        "name": slot["name"],
                        "arguments": arguments,
                    }
                )
            yield {"type": "tool_calls", "tool_calls": assembled}
