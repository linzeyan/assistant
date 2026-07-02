from __future__ import annotations

import json
from pathlib import Path

import pytest

from assistant.models import mlx_engine
from assistant.models.mlx_engine import (
    MlxEnginePool,
    _messages_for_template,
    _render_prompt,
)


class FakeEngine:
    def __init__(self, tag: str):
        self.tag = tag


def _pool(max_loaded: int, pinned=None):
    calls: list[str] = []

    def loader(path: Path, forced_kind: str | None = None) -> FakeEngine:
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


async def test_engine_gets_dedicated_thread_for_its_lifetime():
    # mlx-vlm binds a GPU stream to the thread a model is LOADED on; generating anywhere else
    # raises "There is no Stream(gpu, N) in current thread". The pool therefore owns one
    # single-thread executor per engine: the load runs on it, and executor_for() hands the
    # same thread to generation. Unloading tears it down.
    import threading

    load_threads: dict[str, int] = {}

    def loader(path: Path, forced_kind: str | None = None) -> FakeEngine:
        load_threads[str(path)] = threading.get_ident()
        return FakeEngine(str(path))

    pool = MlxEnginePool(max_loaded=2, loader=loader)
    await pool.acquire("a", Path("/a"))
    executor = pool.executor_for("a")
    assert executor is not None
    gen_thread = executor.submit(threading.get_ident).result()
    assert gen_thread == load_threads["/a"]  # generation lands on the load thread
    await pool.unload("a")
    assert pool.executor_for("a") is None  # torn down with the engine


async def test_no_count_cap_keeps_models_resident_under_ceiling(tmp_path):
    # max_loaded=0 disables count-based eviction: residency is governed by the memory ceiling
    # alone, so several models that fit stay resident (no reload thrash between switches).
    pool = MlxEnginePool(
        max_loaded=0, loader=lambda p, _k=None: FakeEngine(str(p)), mem_ceiling_bytes=10**9
    )
    for name in ("a", "b", "c"):
        d = tmp_path / name
        d.mkdir()
        (d / "w.safetensors").write_bytes(b"\x00" * 100)  # ~100B each, far under ceiling
        await pool.acquire(name, d)
    assert set(pool.loaded_ids()) == {"a", "b", "c"}  # nothing evicted


async def test_no_count_cap_without_ceiling_falls_back_to_one():
    # Uncapped count + no memory ceiling would be unbounded on a machine whose RAM we couldn't
    # detect — the failsafe restores the old single-model behaviour.
    pool = MlxEnginePool(max_loaded=0, loader=lambda p, _k=None: FakeEngine(str(p)))
    await pool.acquire("a", Path("/a"))
    await pool.acquire("b", Path("/b"))
    assert pool.loaded_ids() == ["b"]


async def test_headroom_is_zero_when_count_cap_full(tmp_path):
    # Prefetch asks headroom_bytes() "can another model come in WITHOUT evicting?". A full
    # count cap means the next load evicts regardless of free bytes — so the answer must be 0,
    # or the prefetcher would evict a model that may be mid-generation.
    d = tmp_path / "a"
    d.mkdir()
    (d / "w.safetensors").write_bytes(b"\x00" * 100)
    capped = MlxEnginePool(
        max_loaded=1, loader=lambda p, _k=None: FakeEngine("x"), mem_ceiling_bytes=10**9
    )
    await capped.acquire("a", d)
    assert capped.headroom_bytes() == 0

    uncapped = MlxEnginePool(
        max_loaded=0, loader=lambda p, _k=None: FakeEngine("x"), mem_ceiling_bytes=10**9
    )
    await uncapped.acquire("a", d)
    assert uncapped.headroom_bytes() == 10**9 - 100  # ceiling minus the resident estimate


async def test_resident_acquire_is_not_blocked_by_an_inflight_load():
    # The overlap that makes fusion prefetch worthwhile: while model B loads in the background
    # (holding the pool lock for the whole load), an already-resident model A must still be
    # acquirable instantly — otherwise A's generation queues behind B's load and the "overlap"
    # is actually serialization.
    import asyncio
    import threading
    import time

    gate = threading.Event()

    def loader(path: Path, forced_kind: str | None = None) -> FakeEngine:
        if str(path) == "/slow":
            gate.wait(5)  # hold the pool lock like a real multi-second load
        return FakeEngine(str(path))

    pool = MlxEnginePool(max_loaded=2, loader=loader)
    await pool.acquire("a", Path("/a"))
    slow = asyncio.ensure_future(pool.acquire("slow", Path("/slow")))
    await asyncio.sleep(0.05)  # the slow load is now in flight, lock held
    t0 = time.monotonic()
    engine = await pool.acquire("a", Path("/a"))
    elapsed = time.monotonic() - t0
    gate.set()
    await slow
    assert isinstance(engine, FakeEngine)
    assert elapsed < 0.5  # fast path — did not wait out the slow load


# --- chat-template tool_calls rendering (the "web search just fails" root cause) ---


def _msg_with_args(arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": arguments}}
        ],
    }


def _items_templater(messages, tools=None, add_generation_prompt=True, tokenize=False):
    """Mimics a Qwen3.x template: it iterates each tool call's arguments with jinja
    ``| items``, which raises on a string and only accepts a mapping."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc["function"]["arguments"], dict):
                raise TypeError("Can only get item pairs from a mapping.")
    return "RENDERED"


def test_messages_for_template_parses_string_args_to_dict():
    msgs = [_msg_with_args(json.dumps({"query": "lua http"}))]
    out = _messages_for_template(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"query": "lua http"}
    # Copy-on-write: the persisted history stays string-typed (OpenAI wire format).
    assert isinstance(msgs[0]["tool_calls"][0]["function"]["arguments"], str)


def test_messages_for_template_preserves_unparseable_args():
    msgs = [_msg_with_args("not json")]
    out = _messages_for_template(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_render_prompt_forwards_template_kwargs():
    # Fusion disables thinking via the chat template (Qwen3.x `enable_thinking=False` swaps the
    # open `<think>` for an empty block). The kwargs must reach the templater on BOTH paths —
    # with tools and on the no-tools fallback — or the judge silently reverts to leaking
    # untagged reasoning as its entire answer.
    seen: list[dict] = []

    def templater(messages, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        seen.append(kw)
        return "P"

    _render_prompt(templater, [{"role": "user", "content": "q"}], None,
                   {"enable_thinking": False})
    assert seen[-1] == {"enable_thinking": False}

    def fallback_templater(messages, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        if tools is not None:
            raise TypeError("got an unexpected keyword argument 'tools'")
        seen.append(kw)
        return "P"

    _render_prompt(fallback_templater, [{"role": "user", "content": "q"}], [{"function": {}}],
                   {"enable_thinking": False})
    assert seen[-1] == {"enable_thinking": False}


def test_render_prompt_normalizes_for_items_template():
    # The actual bug: string args + an `| items` template = TypeError. Normalisation fixes it.
    msgs = [_msg_with_args(json.dumps({"query": "lua http"}))]
    assert _render_prompt(_items_templater, msgs, tools=[{"x": 1}]) == "RENDERED"


def test_render_prompt_falls_back_when_tokenizer_rejects_tools_kwarg():
    def picky(messages, add_generation_prompt=True, tokenize=False):  # no `tools` param
        return "RENDERED"

    # First call passes tools= -> TypeError mentioning 'tools' -> retried without it.
    assert _render_prompt(picky, [_msg_with_args("{}")], tools=[{"x": 1}]) == "RENDERED"


def test_render_prompt_surfaces_real_template_error():
    def always_fail(messages, tools=None, add_generation_prompt=True, tokenize=False):
        raise TypeError("Can only get item pairs from a mapping.")

    # A template error unrelated to the tools kwarg must NOT be swallowed by a no-tools retry.
    with pytest.raises(TypeError, match="mapping"):
        _render_prompt(always_fail, [_msg_with_args("{}")], tools=[{"x": 1}])
