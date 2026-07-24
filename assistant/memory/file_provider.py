"""File-backed memory provider.

One append-only JSONL file under the memory dir. Persistence is just the file: a
fresh instance over the same dir sees prior entries.

Search has two modes. With an ``Embedder`` injected (mlx-embeddings installed), each
entry is stored with its embedding and search ranks by cosine similarity — true
semantic recall. Without one, it falls back to keyword-overlap scoring with recency
as the tiebreak. The stored embedding is an implementation detail: it is stripped
from every dict handed back to callers / the API.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .provider import MemoryProvider

if TYPE_CHECKING:
    from assistant.models.mlx_embeddings import Embedder


def _public(entry: dict) -> dict:
    # Hide the stored embedding from callers and the API (it's large and internal).
    return {k: v for k, v in entry.items() if k != "embedding"}


# Cosine floor for semantic recall, measured against the default embedder
# (bge-small-en-v1.5): related pairs score 0.62-0.76 in English but only ~0.51 in
# CJK (the model is English-only), while unrelated pairs span 0.27-0.59. 0.5 keeps
# every measured true hit — no recall regression vs the old unfiltered top-k —
# while dropping the bulk of English-query noise; raising it further has to wait
# for a multilingual embedder.
_MIN_COSINE = 0.5


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class FileMemoryProvider(MemoryProvider):
    def __init__(self, memory_dir: Path, embedder: "Embedder | None" = None):
        self._path = Path(memory_dir) / "memories.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder

    def _semantic_enabled(self) -> bool:
        return self._embedder is not None and self._embedder.available()

    def _load(self) -> list[dict]:
        if not self._path.is_file():
            return []
        entries: list[dict] = []
        for line in self._path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a corrupt line rather than losing all memory
        return entries

    async def write(self, content: str, tags: list[str] | None = None) -> dict:
        entry = {
            "id": uuid.uuid4().hex[:8],
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._semantic_enabled():
            # Embed the content (+ tags) so future queries can match it semantically.
            text = content + (" " + " ".join(tags) if tags else "")
            try:
                vectors = await self._embedder.embed([text])
                if vectors:
                    entry["embedding"] = vectors[0]
            except Exception:
                # Never let an embedding failure lose the memory itself.
                pass
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return _public(entry)

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        if not query.strip():
            return []
        if self._semantic_enabled():
            hits = await self._semantic_search(query, limit)
            if hits is not None:
                # Union in exact-term matches: they bridge what the embedder misses
                # (identifiers, CJK under the English-only default model).
                if len(hits) < limit:
                    seen = {h["id"] for h in hits}
                    extra = [
                        h
                        for h in self._keyword_search(query, limit)
                        if h["id"] not in seen
                    ]
                    hits += extra[: limit - len(hits)]
                return hits
        return self._keyword_search(query, limit)

    async def _semantic_search(self, query: str, limit: int) -> list[dict] | None:
        entries = [e for e in self._load() if isinstance(e.get("embedding"), list)]
        if not entries:
            return None  # nothing embedded yet -> let keyword search handle it
        try:
            qvec = (await self._embedder.embed([query]))[0]
        except Exception:
            return None
        scored = [(_cosine(qvec, e["embedding"]), e) for e in entries]
        # Below-floor entries are noise, not context: injecting them every turn is
        # how memory becomes a garbage dump. Ties go to the newer entry so a
        # superseding fact outranks the stale one it replaces.
        relevant = [(s, e) for s, e in scored if s >= _MIN_COSINE]
        relevant.sort(key=lambda t: (t[0], t[1].get("created_at", "")), reverse=True)
        return [_public(e) for _, e in relevant[:limit]]

    def _keyword_search(self, query: str, limit: int) -> list[dict]:
        terms = query.lower().split()
        if not terms:
            return []
        entries = self._load()
        scored: list[tuple[int, int, dict]] = []
        for idx, e in enumerate(entries):  # idx encodes recency (higher == newer)
            haystack = (e.get("content", "") + " " + " ".join(e.get("tags", []))).lower()
            score = sum(1 for t in terms if t in haystack)
            if score:
                scored.append((score, idx, e))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [_public(e) for _, _, e in scored[:limit]]

    async def all(self) -> list[dict]:
        return [_public(e) for e in reversed(self._load())]  # most recent first

    async def update(
        self, entry_id: str, content: str, tags: list[str] | None = None
    ) -> dict | None:
        entries = self._load()
        target = next((e for e in entries if e.get("id") == entry_id), None)
        if target is None:
            return None
        target["content"] = content
        target["tags"] = tags or []
        target.pop("embedding", None)  # content changed — the old vector is stale
        if self._semantic_enabled():
            text = content + (" " + " ".join(tags) if tags else "")
            try:
                vectors = await self._embedder.embed([text])
                if vectors:
                    target["embedding"] = vectors[0]
            except Exception:
                pass  # an embedding failure must not block the edit
        self._rewrite(entries)
        return _public(target)

    async def delete(self, entry_id: str) -> bool:
        entries = self._load()
        kept = [e for e in entries if e.get("id") != entry_id]
        if len(kept) == len(entries):
            return False
        self._rewrite(kept)
        return True

    def _rewrite(self, entries: list[dict]) -> None:
        # The append-only file is the fast path; edit/delete pay a full rewrite. Write to
        # a temp file then atomically replace so a crash mid-write can't truncate memory.
        tmp = self._path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        tmp.replace(self._path)
