"""Semantic memory search via an injected embedder (faked for determinism)."""

from __future__ import annotations

from assistant.memory.file_provider import FileMemoryProvider

_VOCAB = ["apple", "banana", "cherry", "deploy"]


class FakeEmbedder:
    """Deterministic bag-of-vocab vectors so cosine ranking is predictable."""

    def __init__(self, available: bool = True):
        self._available = available

    def available(self) -> bool:
        return self._available

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(t.lower().count(w)) for w in _VOCAB] for t in texts]


async def test_semantic_search_ranks_by_meaning(tmp_path):
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    await m.write("apple orchard notes")
    await m.write("banana bread recipe")
    hits = await m.search("apple")
    assert hits[0]["content"] == "apple orchard notes"
    assert "embedding" not in hits[0]  # internal field never leaks to callers


async def test_write_persists_embedding_but_hides_it(tmp_path):
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    entry = await m.write("apple")
    assert "embedding" not in entry
    assert "embedding" in (tmp_path / "memories.jsonl").read_text()


async def test_falls_back_to_keyword_without_embedder(tmp_path):
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder(available=False))
    await m.write("python testing with pytest")
    await m.write("python is a language")
    hits = await m.search("python pytest testing")
    assert hits[0]["content"] == "python testing with pytest"


async def test_semantic_falls_back_when_entries_have_no_embeddings(tmp_path):
    # Written before any embedder existed...
    await FileMemoryProvider(tmp_path).write("deploy runbook")
    # ...then searched once an embedder is available -> keyword fallback still finds it.
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    hits = await m.search("deploy")
    assert hits and hits[0]["content"] == "deploy runbook"


async def test_irrelevant_memories_are_not_injected(tmp_path):
    # Below the cosine floor nothing is returned: without it every prefetch shipped
    # the top-k regardless of relevance, so unrelated turns got memory noise anyway.
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    await m.write("apple orchard notes")
    await m.write("banana bread recipe")
    assert await m.search("cherry") == []
    assert await m.prefetch("cherry") == ""


async def test_semantic_tie_prefers_newer_entry(tmp_path):
    # Equal similarity must rank the newer fact first — that is what lets a
    # superseding memory shadow the stale one it replaces.
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    await m.write("apple fact old")
    await m.write("apple fact new")
    hits = await m.search("apple")
    assert hits[0]["content"] == "apple fact new"


async def test_keyword_union_bridges_semantic_gap(tmp_path):
    # "zap-cli" is outside the fake embedder's vocab (cosine 0, below the floor),
    # but an exact-term match must still recall it — identifiers and CJK under the
    # English-only default embedder depend on this bridge.
    m = FileMemoryProvider(tmp_path, embedder=FakeEmbedder())
    await m.write("deploy with zap-cli runbook")
    hits = await m.search("zap-cli")
    assert hits and hits[0]["content"] == "deploy with zap-cli runbook"


async def test_embedding_failure_never_loses_the_memory(tmp_path):
    class Boom(FakeEmbedder):
        async def embed(self, texts):
            raise RuntimeError("embed boom")

    m = FileMemoryProvider(tmp_path, embedder=Boom())
    entry = await m.write("kept anyway")
    assert entry["content"] == "kept anyway"
    assert any(e["content"] == "kept anyway" for e in await m.all())
