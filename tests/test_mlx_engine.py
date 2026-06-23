from __future__ import annotations

from pathlib import Path

from assistant.models import mlx_engine
from assistant.models.mlx_engine import MlxEnginePool


class FakeEngine:
    def __init__(self, tag: str):
        self.tag = tag


def _pool(max_loaded: int, pinned=None):
    calls: list[str] = []

    def loader(path: Path) -> FakeEngine:
        calls.append(str(path))
        return FakeEngine(str(path))

    return MlxEnginePool(max_loaded=max_loaded, loader=loader, pinned=pinned), calls


async def test_acquire_loads_once_then_reuses():
    pool, calls = _pool(max_loaded=2)
    a1 = await pool.acquire("a", Path("/m/a"))
    a2 = await pool.acquire("a", Path("/m/a"))
    assert a1 is a2  # reused, not reloaded
    assert calls == ["/m/a"]
    assert pool.loaded_ids() == ["a"]


async def test_lru_eviction_when_over_budget():
    pool, _ = _pool(max_loaded=2)
    await pool.acquire("a", Path("/a"))
    await pool.acquire("b", Path("/b"))
    await pool.acquire("c", Path("/c"))  # evicts LRU "a"
    assert set(pool.loaded_ids()) == {"b", "c"}
    assert not pool.is_loaded("a")


async def test_reacquire_marks_most_recently_used():
    pool, _ = _pool(max_loaded=2)
    await pool.acquire("a", Path("/a"))
    await pool.acquire("b", Path("/b"))
    await pool.acquire("a", Path("/a"))  # "a" becomes MRU, "b" now LRU
    await pool.acquire("c", Path("/c"))  # evicts "b"
    assert set(pool.loaded_ids()) == {"a", "c"}


async def test_pinned_model_is_not_evicted():
    pool, _ = _pool(max_loaded=1, pinned={"a"})
    await pool.acquire("a", Path("/a"))
    await pool.acquire("b", Path("/b"))  # cannot evict pinned "a" -> exceeds budget
    assert pool.is_loaded("a")
    assert pool.is_loaded("b")


async def test_unload_reports_whether_present():
    pool, _ = _pool(max_loaded=2)
    await pool.acquire("a", Path("/a"))
    assert await pool.unload("a") is True
    assert await pool.unload("a") is False
    assert pool.loaded_ids() == []


async def test_unload_releases_mlx_memory(monkeypatch):
    # Unloading must clear MLX's Metal buffer cache, not just drop the dict ref —
    # otherwise unified memory isn't returned. A no-op unload must NOT pay that cost.
    calls = {"n": 0}
    monkeypatch.setattr(
        mlx_engine, "_release_mlx_memory", lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    pool, _ = _pool(max_loaded=2)
    await pool.acquire("a", Path("/a"))
    await pool.unload("a")
    assert calls["n"] == 1
    await pool.unload("a")  # already gone — nothing to release
    assert calls["n"] == 1


async def test_eviction_releases_mlx_memory(monkeypatch):
    # Eviction during acquire leaks the same way unless it clears the cache too.
    calls = {"n": 0}
    monkeypatch.setattr(
        mlx_engine, "_release_mlx_memory", lambda: calls.__setitem__("n", calls["n"] + 1)
    )
    pool, _ = _pool(max_loaded=1)
    await pool.acquire("a", Path("/a"))
    await pool.acquire("b", Path("/b"))  # evicts "a"
    assert calls["n"] >= 1
