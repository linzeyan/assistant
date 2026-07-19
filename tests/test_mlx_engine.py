from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from assistant.models import mlx_engine
from assistant.models.mlx_engine import (
    MlxEngine,
    MlxEnginePool,
    _common_prefix_len,
    _messages_for_template,
    _normalize_message_shape,
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
    # After the shape-normalisation retry ALSO fails, the ORIGINAL error is what surfaces.
    with pytest.raises(TypeError, match="mapping"):
        _render_prompt(always_fail, [_msg_with_args("{}")], tools=[{"x": 1}])


class _TemplateReject(Exception):
    """Stands in for jinja2's TemplateError raised from inside a strict chat template."""


# Claude Code produces these shapes; strict templates reject them and the panel member was skipped.
_CC_SHAPE = [
    {"role": "system", "content": "You are Claude Code."},
    {"role": "user", "content": "read a.py"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "read_file", "arguments": {"path": "a.py"}}}]},
    {"role": "tool", "content": "file contents"},
    {"role": "user", "content": "now edit it"},
    {"role": "system", "content": "<system-reminder> stay on task"},
    {"role": "user", "content": "go"},
]


def _alternation_strict(messages, tools=None, add_generation_prompt=True, tokenize=False):
    """Mixtral: only user/assistant, and roles must strictly alternate."""
    prev = None
    for m in messages:
        role = m["role"]
        if role not in ("user", "assistant", "system"):
            raise _TemplateReject("Conversation roles must alternate user/assistant/…")
        if role == "system":
            continue
        if role == prev:
            raise _TemplateReject("Conversation roles must alternate user/assistant/…")
        prev = role
    return "RENDERED"


def _system_first_strict(messages, tools=None, add_generation_prompt=True, tokenize=False):
    """Qwen3.x: a system message may appear only at index 0."""
    for i, m in enumerate(messages):
        if m["role"] == "system" and i != 0:
            raise _TemplateReject("System message must be at the beginning of the conversation.")
    return "RENDERED"


def test_render_prompt_normalizes_for_alternation_strict_template():
    # Mixtral-class crash ("roles must alternate"): tool role + consecutive users. Normalisation
    # folds the tool result in and merges consecutive turns so the retry renders.
    assert _render_prompt(_alternation_strict, _CC_SHAPE, tools=None) == "RENDERED"


def test_render_prompt_normalizes_for_system_first_strict_template():
    # Qwen3.x crash ("system must be at the beginning"): the mid-conversation <system-reminder>
    # gets hoisted/merged into the single leading system message.
    assert _render_prompt(_system_first_strict, _CC_SHAPE, tools=None) == "RENDERED"


def test_render_prompt_does_not_normalize_when_template_accepts_shape():
    # A tool-aware template that renders the original must keep the EXACT (structured) messages —
    # normalisation is a failure-path-only fallback and must never touch the happy path.
    seen: list[list[dict]] = []

    def tolerant(messages, tools=None, add_generation_prompt=True, tokenize=False):
        seen.append(messages)
        return "OK"

    assert _render_prompt(tolerant, _CC_SHAPE, tools=None) == "OK"
    assert len(seen) == 1  # rendered once, no retry
    assert any(m.get("tool_calls") for m in seen[0])  # tool structure preserved, not flattened


def test_normalize_message_shape_hoists_system_folds_tools_merges_turns():
    out = _normalize_message_shape(_CC_SHAPE)
    # Exactly one system message, and it leads.
    assert out[0]["role"] == "system"
    assert [m["role"] for m in out].count("system") == 1
    assert "stay on task" in out[0]["content"]  # mid-list system merged in
    # Only user/assistant after the system, strictly alternating.
    roles = [m["role"] for m in out[1:]]
    assert all(r in ("user", "assistant") for r in roles)
    assert all(a != b for a, b in zip(roles, roles[1:]))
    # The tool call survives as text on the assistant turn; the tool result folded into a user turn.
    assert any("[called: read_file(" in m["content"] for m in out if m["role"] == "assistant")
    assert any("file contents" in m["content"] for m in out if m["role"] == "user")


def test_normalize_message_shape_noop_shape_is_unchanged():
    # A list that already satisfies the strict rules round-trips to the same content.
    clean = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _normalize_message_shape(clean) == clean


# --- prompt/KV cache reuse across turns (B) ---


def test_common_prefix_len():
    assert _common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert _common_prefix_len([1, 2], [1, 2, 3]) == 2  # shorter bounds it
    assert _common_prefix_len([], [1]) == 0
    assert _common_prefix_len([5, 6], [7, 8]) == 0


class _StubTok:
    """A tokenizer whose encode() returns a settable id list, so a test can script two turns'
    prompts and control the prefix overlap between them."""

    bos_token = None

    def __init__(self, ids):
        self.ids = list(ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True,
                            tokenize=False, **kw):
        return "RENDERED"

    def encode(self, text, add_special_tokens=True):
        return list(self.ids)


class _FakeCache:
    """Distinguishable stand-in for an mlx-lm prompt cache (identity is what the tests check)."""


def _install_mlx(monkeypatch, *, script, trim_result=None, raise_after=None):
    """Stub mlx_lm.stream_generate + the cache helpers (the real package can't import in CI). Records
    each stream_generate call's (prompt, cache) so a test can assert what got prefilled and reused."""
    calls: list[dict] = []

    def stream_generate(model, tokenizer, prompt, max_tokens=256, prompt_cache=None, **kw):
        calls.append({"prompt": list(prompt), "cache": prompt_cache, "max_tokens": max_tokens})
        for i, (tid, text) in enumerate(script):
            if raise_after is not None and i == raise_after:
                raise RuntimeError("boom")
            yield types.SimpleNamespace(token=tid, text=text)

    def make_prompt_cache(model, max_kv_size=None):
        return _FakeCache()

    def trim_prompt_cache(cache, n):
        return n if trim_result is None else trim_result

    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.stream_generate = stream_generate
    models = types.ModuleType("mlx_lm.models")
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = make_prompt_cache
    cache_mod.trim_prompt_cache = trim_prompt_cache
    models.cache = cache_mod
    mlx_lm.models = models
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_mod)
    return calls


def test_stream_text_first_turn_prefills_full_and_reports_usage(monkeypatch):
    calls = _install_mlx(monkeypatch, script=[(10, "a"), (11, "b")])
    eng = MlxEngine(model=object(), tokenizer=_StubTok([1, 2, 3]))
    usage: dict = {}
    out = list(eng.stream_text([{"role": "user", "content": "hi"}], usage_out=usage))
    assert out == ["a", "b"]
    assert calls[0]["prompt"] == [1, 2, 3]  # no prior cache → whole prompt prefilled
    assert usage == {"input_tokens": 3, "output_tokens": 2}
    assert eng._cache_ids == [1, 2, 3, 10, 11]  # prompt + generated
    assert eng._cache is calls[0]["cache"]  # the fresh cache was committed


def test_stream_text_reuses_cache_on_shared_prefix(monkeypatch):
    tok = _StubTok([1, 2, 3])
    calls = _install_mlx(monkeypatch, script=[(10, "a"), (11, "b")])
    eng = MlxEngine(model=object(), tokenizer=tok)
    list(eng.stream_text([{"role": "user", "content": "hi"}]))  # cache_ids -> [1,2,3,10,11]
    first_cache = eng._cache
    tok.ids = [1, 2, 3, 10, 11, 20]  # next turn = full prior context + one new token
    list(eng.stream_text([{"role": "user", "content": "more"}]))
    assert calls[1]["prompt"] == [20]  # ONLY the new tail is prefilled
    assert calls[1]["cache"] is first_cache  # against the retained cache


def test_stream_text_trims_cache_to_shared_prefix(monkeypatch):
    tok = _StubTok([1, 2, 3])
    calls = _install_mlx(monkeypatch, script=[(10, "a"), (11, "b")])  # trim returns n (full)
    eng = MlxEngine(model=object(), tokenizer=tok)
    list(eng.stream_text([{"role": "user", "content": "hi"}]))  # cache_ids -> [1,2,3,10,11]
    first_cache = eng._cache
    tok.ids = [1, 2, 3, 99]  # diverges after [1,2,3]; cache must drop its last 2 tokens
    list(eng.stream_text([{"role": "user", "content": "x"}]))
    assert calls[1]["prompt"] == [99]
    assert calls[1]["cache"] is first_cache  # trimmed back to the shared prefix and reused


def test_stream_text_rebuilds_when_cache_cannot_be_trimmed(monkeypatch):
    tok = _StubTok([1, 2, 3])
    calls = _install_mlx(monkeypatch, script=[(10, "a")], trim_result=0)  # trim can't drop → 0
    eng = MlxEngine(model=object(), tokenizer=tok)
    list(eng.stream_text([{"role": "user", "content": "hi"}]))  # cache_ids -> [1,2,3,10]
    first_cache = eng._cache
    tok.ids = [1, 2, 3, 99]  # needs a 1-token trim, which the cache refuses
    list(eng.stream_text([{"role": "user", "content": "x"}]))
    assert calls[1]["prompt"] == [1, 2, 3, 99]  # fall back to a full prefill
    assert calls[1]["cache"] is not first_cache  # on a fresh cache


def test_stream_text_logs_prefill_vs_decode_split(monkeypatch, caplog):
    # The observability contract for "why is this slow": the log line must separate cached vs
    # prefilled prompt tokens (cache health) from decode count (model ceiling).
    tok = _StubTok([1, 2, 3])
    _install_mlx(monkeypatch, script=[(10, "a"), (11, "b")])
    eng = MlxEngine(model=object(), tokenizer=tok)
    with caplog.at_level("INFO", logger="assistant"):
        list(eng.stream_text([{"role": "user", "content": "hi"}]))
        tok.ids = [1, 2, 3, 10, 11, 20]  # second turn: cache hit on all but 1 token
        list(eng.stream_text([{"role": "user", "content": "more"}]))
    lines = [r.message for r in caplog.records if r.message.startswith("generation:")]
    assert "prompt=3 (cached=0, prefill=3)" in lines[0]  # cold cache: everything prefilled
    assert "prompt=6 (cached=5, prefill=1)" in lines[1]  # warm: only the new tail
    assert "decode=2 tok" in lines[0]


def test_stream_text_invalidates_cache_on_error(monkeypatch):
    _install_mlx(monkeypatch, script=[(10, "a"), (11, "b")], raise_after=1)
    eng = MlxEngine(model=object(), tokenizer=_StubTok([1, 2, 3]))
    with pytest.raises(RuntimeError):
        list(eng.stream_text([{"role": "user", "content": "hi"}]))
    # A mid-stream failure must not leave a cache whose token record is out of sync.
    assert eng._cache is None and eng._cache_ids == []


def _install_mlx_vlm(monkeypatch, processor):
    """Stub mlx_vlm so _load_vlm can run in CI (mlx_vlm isn't importable there)."""
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.load = lambda path: (object(), processor)
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)


def test_load_vlm_refuses_checkpoint_without_chat_template(monkeypatch):
    # A template-less checkpoint must fail AT LOAD: mlx-vlm's apply_chat_template silently
    # renders bare "System:/User:" text (no turn tokens, no tools, no stop discipline) and
    # the model rambles for tens of thousands of tokens per turn (N80, gemma-4-12B-bf16).
    processor = types.SimpleNamespace(
        chat_template=None, tokenizer=types.SimpleNamespace(chat_template=None)
    )
    _install_mlx_vlm(monkeypatch, processor)
    with pytest.raises(RuntimeError, match="no chat template"):
        mlx_engine._load_vlm(Path("/m/broken"))


def test_load_vlm_accepts_checkpoint_with_chat_template(monkeypatch):
    # Template on the processor OR its tokenizer is fine — both are where mlx-vlm puts it.
    processor = types.SimpleNamespace(
        chat_template=None,
        tokenizer=types.SimpleNamespace(chat_template="{{ messages }}"),
    )
    _install_mlx_vlm(monkeypatch, processor)
    engine = mlx_engine._load_vlm(Path("/m/ok"))
    assert engine._processor is processor
