from __future__ import annotations

from abc import ABC, abstractmethod


class MemoryProvider(ABC):
    """The single pluggable seam for long-term memory.

    Deliberately tiny — only the load-bearing methods from hermes-agent's provider
    ABC. We ship exactly ONE implementation (file-based); the abstraction exists so a
    future vector/DB backend can drop in, NOT so eight providers can compete.
    """

    @abstractmethod
    async def write(self, content: str, tags: list[str] | None = None) -> dict: ...

    @abstractmethod
    async def search(self, query: str, limit: int = 5) -> list[dict]: ...

    @abstractmethod
    async def all(self) -> list[dict]: ...

    async def prefetch(self, query: str) -> str:
        """Return a formatted block of relevant memories to inject before an LLM
        call, or '' if nothing is relevant. Default: top matches as a bullet list."""
        hits = await self.search(query, limit=5)
        return "\n".join(f"- {h['content']}" for h in hits)

    async def sync_turn(self, user: str, assistant: str) -> None:
        """Optional post-turn hook. No-op by default: memory is written explicitly by
        the agent via the memory_write tool, not auto-extracted (keeps it signal, not
        noise)."""
        return None
