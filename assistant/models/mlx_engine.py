"""In-memory engine pool for the native MLX backend.

This is the project's own version of omlx's ``EnginePool`` — the "preload / unload /
switch" management the user wants, owned natively so it needs no external omlx
server. The pool keeps loaded ``(model, tokenizer)`` engines in memory, evicts the
least-recently-used one when over budget, supports manual load/unload, and can pin
models so a hot model is never evicted.

Loading and generation are blocking (MLX runs on the CPU/GPU synchronously), so the
pool loads via ``asyncio.to_thread`` and callers stream generation off the event
loop (see ``MlxModelService``). The actual ``mlx_lm`` import is deferred to the
default loader so this module imports fine on machines without MLX installed, and so
tests can inject a fake loader.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .tool_parsing import HARMONY_CHANNEL, harmony_fields

log = logging.getLogger("assistant")


def _release_mlx_memory() -> None:
    """Return a just-unloaded model's unified memory to the system.

    Dropping the last Python reference to an mlx model is necessary but not sufficient:
    MLX keeps freed buffers in a Metal allocator cache, so resident memory doesn't fall
    until that cache is cleared. A GC pass finalises the now-unreferenced model, then
    ``mx.clear_cache()`` releases the pooled buffers. Import-guarded so it's a no-op on
    machines without MLX (and under the fake-loader tests), and tolerant of the older
    ``mx.metal.clear_cache`` spelling.
    """
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(
        getattr(mx, "metal", None), "clear_cache", None
    )
    if callable(clear):
        clear()
    # Log the active/cache footprint so "unload didn't free memory" reports are diagnosable:
    # active = memory still held by live arrays (a leaked reference shows up here), cache =
    # pooled-but-free buffers (should drop to ~0 right after clear_cache).
    active = getattr(mx, "get_active_memory", None)
    cache = getattr(mx, "get_cache_memory", None)
    if callable(active) and callable(cache):
        log.info(
            "mlx memory after release: active=%.2fGB cache=%.2fGB",
            active() / 1e9, cache() / 1e9,
        )


# MLX keeps its compiled-graph cache in THREAD-LOCAL storage, and mlx-lm compiles on every
# generation (its sampler is ``@mx.compile``d), so every model thread ends up holding one. When such
# a thread exits, that cache's C++ destructor runs from a pthread TSD callback and frees Python
# objects with no GIL held — racing whatever the event loop is doing. We caught it exactly that way:
# a crash report shows the dying thread in ``CompilerCache::~CompilerCache -> tupledealloc`` while
# the main thread sat in ``gc_collect`` (our own ``_release_mlx_memory``, called microseconds after
# ``_drop`` shut the thread down), and the backend took a SIGSEGV. MLX exposes no Python API to clear
# this cache, so we call the C++ symbol directly — ON that thread, before it is allowed to exit.
_COMPILE_CACHE_CLEAR_SYMBOL = "_ZN3mlx4core6detail19compile_clear_cacheEv"
_UNRESOLVED = object()
# Resolved once, lazily. No lock: the GIL makes the store atomic and resolving twice is harmless.
_compile_cache_clear: object = _UNRESOLVED
# Threads we must never let exit because MLX is loaded but its clear symbol is gone (see
# ``_retire_executor``). A leaked idle thread is cheap; the alternative is a crashed backend.
_immortal_executors: list[ThreadPoolExecutor] = []


def _compile_cache_cleaner():
    """MLX's thread-local compile-cache clear, or None when it can't be called.

    ``ctypes.PyDLL``, not ``CDLL``: the clear decrefs Python objects, so it must run with the GIL
    held — which is the whole point of doing this on purpose instead of leaving it to thread exit.
    """
    global _compile_cache_clear
    if _compile_cache_clear is not _UNRESOLVED:
        return _compile_cache_clear
    fn = None
    try:
        import ctypes

        import mlx.core as mx

        lib = Path(mx.__file__).parent / "lib" / "libmlx.dylib"
        fn = getattr(ctypes.PyDLL(str(lib)), _COMPILE_CACHE_CLEAR_SYMBOL, None)
    except Exception:  # MLX absent, or the library moved — both mean "can't clear"
        fn = None
    if fn is None and _mlx_importable():
        log.warning(
            "mlx compile-cache clear (%s) not found in libmlx — model threads will be kept alive "
            "instead of exiting, to avoid the thread-local teardown crash",
            _COMPILE_CACHE_CLEAR_SYMBOL,
        )
    _compile_cache_clear = fn
    return fn


def _clear_thread_compile_cache() -> None:
    """Clear the calling thread's MLX compile cache. Submitted as a model thread's LAST job.

    Module-level and zero-argument on purpose: a closure or bound method would keep whatever it
    captured alive until the thread drains, which is exactly what
    ``test_unload_frees_engine_not_pinned_by_stream_frame`` forbids.
    """
    clear = _compile_cache_cleaner()
    if clear is not None:
        clear()


def _mlx_importable() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _retire_executor(executor: ThreadPoolExecutor) -> None:
    """Retire a model's dedicated thread without triggering the MLX teardown crash.

    The clear is submitted as ordinary work, so it queues *behind* any in-flight generation and is
    provably the last thing that thread runs; ``shutdown(wait=False)`` then returns immediately, so
    the event loop is never blocked. (omlx solves the same crash by blocking on the clear with a
    timeout and calling ``os._exit`` if it expires — it can afford that behind a supervisor; here it
    would freeze the backend for the length of a turn.)
    """
    if _compile_cache_cleaner() is not None:
        executor.submit(_clear_thread_compile_cache)
        executor.shutdown(wait=False)
    elif not _mlx_importable():
        # Nothing was ever compiled on it (fake-loader tests / non-MLX machines) — and taking the
        # immortal path here would leak a thread per model in every test run.
        executor.shutdown(wait=False)
    else:
        _immortal_executors.append(executor)


def _total_ram_bytes() -> int | None:
    """Physical RAM in bytes, or None if it can't be determined. Used to default the pool's
    admission ceiling to the machine's actual memory so an oversized model fails loud with a clear
    message instead of OOM-crashing the backend — "check the resource fits before loading"."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")  # macOS + Linux
    except (ValueError, OSError, AttributeError):
        return None


def _active_memory_bytes() -> int | None:
    """Current MLX unified-memory footprint in bytes, or None when MLX isn't importable
    (fake-loader tests / non-MLX machines). Lets the pool refine a model's disk-size estimate
    with its real post-load footprint for the admission ceiling (see ``MlxEnginePool.acquire``)."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    active = getattr(mx, "get_active_memory", None)
    return int(active()) if callable(active) else None


def _peak_memory_bytes() -> int | None:
    """MLX's high-water unified-memory mark since the last reset, or None when MLX isn't
    importable. Paired with ``_reset_peak_memory`` to measure what one turn of generation really
    costs on top of the resident weights (see ``MlxEngine.working_memory_bytes``)."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    peak = getattr(mx, "get_peak_memory", None)
    return int(peak()) if callable(peak) else None


def _reset_peak_memory() -> None:
    try:
        import mlx.core as mx
    except ImportError:
        return
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def _estimate_model_bytes(path: Path) -> int:
    """Estimate a model's resident footprint from its on-disk weight shards.

    The summed ``*.safetensors`` size is the dominant term (the weights) and the only signal
    available *before* a load. It UNDER-estimates — runtime adds a KV cache and activations that
    grow with context — so it's used only as the pre-load admission guess and is replaced by the
    measured active-memory delta once the model is in. Returns 0 when no weight files are found,
    which disables byte-admission for that model rather than guessing from nothing.
    """
    try:
        return sum(f.stat().st_size for f in path.glob("*.safetensors"))
    except OSError:
        return 0


# Weights are only part of what a resident model costs: generation adds a KV cache and
# activations that no pre-load signal can see. Booking weights alone is what let three models
# total 114GB "fit" under a 124GB ceiling and then OOM-kill the backend on the first decode.
# Until a model has actually generated — at which point the pool books its measured working
# memory instead — budget this multiple of its weights. Proportional rather than a flat reserve
# because the KV cache scales with the model, and deliberately coarse: it only has to stop an
# admission that would leave no room to decode.
_WORKING_MEMORY_FACTOR = 1.2


class ModelAdmissionError(RuntimeError):
    """A model can't be admitted under the configured memory ceiling. Raised *before* the load so
    a too-big model fails with a clear message (surfaced as a chat error / a 502 on /models load)
    instead of letting MLX OOM-crash the whole backend."""


# The N84 stream sanitizer emits harmony reasoning in the product's ``<think>`` display
# convention (N1), so NEW history arrives think-wrapped; OLD sessions still carry raw
# <|channel|> markup. Both must map back to the template's thinking/content fields —
# gpt-oss's template rejects <|channel|> in content (N82) and would replay <think> text
# as final-channel prose.
_THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)


def _split_preopened_think(content: str) -> tuple[str, str] | None:
    """``(reasoning, answer)`` where the model was already inside a think block when it started
    generating, else None.

    Qwen3.x's template ends its generation prompt with a bare ``<think>``, so the reply opens
    inside the block and the only tag the model ever emits is the *closing* one. Nothing that
    looks for a ``<think>`` finds anything, so the whole scratchpad is kept as the answer: the
    caller is handed the model's reasoning with a stray ``</think>`` in the middle of it, and —
    worse — the next render replays that text as prose, because the template puts reasoning in
    its own ``reasoning_content`` field and this never reached it. Every later turn then carries
    every earlier turn's thinking, in the one form the template cannot strip.
    """
    if "<think>" in content or "</think>" not in content:
        return None
    reasoning, _, answer = content.partition("</think>")
    return reasoning.strip(), answer.strip()


def _messages_for_template(messages: list[dict], harmony: bool = False) -> list[dict]:
    """Return messages reshaped for chat-template rendering only: assistant tool_calls'
    ``arguments`` parsed from JSON string to a dict, and (for harmony models) reasoning
    text split back into ``thinking``/``content`` fields.

    Sessions persist tool_calls in OpenAI wire format — ``arguments`` is a JSON *string* — but
    HF chat templates expect a parsed mapping: Qwen3.x iterates it with jinja ``| items``,
    which raises "Can only get item pairs from a mapping" on a string (the real cause of the
    "web search just fails" reports). Copy-on-write so the persisted history stays
    string-typed; a value that doesn't parse is left untouched.
    """
    out: list[dict] = []
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and isinstance(m.get("content"), str)
        ):
            if HARMONY_CHANNEL in m["content"]:  # pre-N84 history: raw channel markup
                thinking, content = harmony_fields(m["content"])
                m = {**m, "content": content}
                if thinking:
                    m["thinking"] = thinking
            elif harmony and (tm := _THINK_RE.match(m["content"])):
                m = {**m, "content": tm.group(2).strip()}
                if tm.group(1).strip():
                    m["thinking"] = tm.group(1).strip()
            elif (split := _split_preopened_think(m["content"])) is not None:
                # ``reasoning_content`` is the field Qwen3.x's template reads, and it is the
                # template's own decision whether to replay it — which is the point: reasoning
                # left in ``content`` is replayed unconditionally and forever.
                m = {**m, "content": split[1]}
                if split[0]:
                    m["reasoning_content"] = split[0]
        tcs = m.get("tool_calls") if isinstance(m, dict) else None
        if not tcs:
            out.append(m)
            continue
        new_tcs = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = json.loads(args)
                except (ValueError, TypeError):
                    new_tcs.append(tc)  # not JSON — leave as-is, template may accept strings
                    continue
                new_tcs.append({**tc, "function": {**fn, "arguments": parsed}})
            else:
                new_tcs.append(tc)
        out.append({**m, "tool_calls": new_tcs})
    return out


def _render_prompt(
    templater,
    messages: list[dict],
    tools: list[dict] | None,
    template_kwargs: dict | None = None,
    harmony: bool = False,
) -> str:
    """Render the chat prompt, normalising tool_calls first and falling back ONLY when the
    tokenizer genuinely rejects the ``tools`` kwarg.

    A TypeError from *inside* the template (a message-shape mismatch) must surface — retrying
    without tools would just fail the same way and mask the real cause. The previous blanket
    ``except TypeError`` swallowed exactly that, hiding the Qwen3.x tool_calls render bug.

    ``template_kwargs`` are forwarded into the chat template's jinja context (e.g. Qwen3.x's
    ``enable_thinking=False``, which swaps the generation prompt's open ``<think>`` for an empty
    block so the model answers directly). Templates that don't know a variable ignore it.
    """
    messages = _messages_for_template(messages, harmony=harmony)
    extra = template_kwargs or {}
    try:
        return _apply_template(templater, messages, tools, extra)
    except Exception as original:
        # A strict chat template rejected Claude Code's message shape — Mixtral demands strict
        # user/assistant alternation ("roles must alternate"), Qwen3.x demands the system message
        # come first ("System message must be at the beginning"). Both fire from inside jinja as a
        # TemplateError, not a TypeError, so the tools-kwarg fallback below never caught them and the
        # panel member was silently skipped. Normalise to the strict common denominator and retry
        # ONCE. Models whose template accepted the original never reach here, so they keep their exact
        # (tool-aware) message shape — only the already-failing path pays for the rewrite.
        normalized = _normalize_message_shape(messages)
        if normalized == messages:
            raise
        try:
            return _apply_template(templater, normalized, tools, extra)
        except Exception:
            raise original from None


def _apply_template(templater, messages: list[dict], tools, extra: dict) -> str:
    """Render once, falling back only when the tokenizer genuinely rejects the ``tools`` kwarg.

    A TypeError from *inside* the template (a message-shape mismatch) must surface — retrying
    without tools would just fail the same way and mask the real cause. The previous blanket
    ``except TypeError`` swallowed exactly that, hiding the Qwen3.x tool_calls render bug.
    """
    try:
        return templater(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, **extra
        )
    except TypeError as exc:
        if "tools" not in str(exc):  # not the "template doesn't accept tools" case — surface it
            raise
        return templater(messages, add_generation_prompt=True, tokenize=False, **extra)


def _normalize_message_shape(messages: list[dict]) -> list[dict]:
    """Rewrite a message list to the strict common denominator strict chat templates demand: a
    single leading system message, then only user/assistant turns that alternate. Called ONLY as a
    fallback after a template rejected the original shape, so well-behaved (tool-aware) templates are
    never touched.

    Because a template this strict has no tool slot anyway, tool structure is flattened to text: a
    ``role: tool`` result folds into the conversation as a user turn, and an assistant's
    ``tool_calls`` are appended to its text so the history isn't lost. Consecutive same-role turns
    are then merged to satisfy alternation.
    """
    system_parts: list[str] = []
    convo: list[tuple[str, str]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "assistant" and m.get("tool_calls"):
            rendered = "; ".join(
                f"{(tc.get('function') or {}).get('name', '')}"
                f"({(tc.get('function') or {}).get('arguments', '')})"
                for tc in m["tool_calls"]
            )
            content = f"{content}\n[called: {rendered}]".strip()
        elif role == "tool":  # no tool slot in a strict template — fold the result in as a user turn
            role = "user"
        if role not in ("user", "assistant"):
            role = "user"
        convo.append((role, content))

    merged: list[list] = []
    for role, content in convo:
        if merged and merged[-1][0] == role:
            merged[-1][1] = f"{merged[-1][1]}\n\n{content}".strip()
        else:
            merged.append([role, content])

    out: list[dict] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    out.extend({"role": r, "content": c} for r, c in merged)
    return out


def _sampler_kwargs(
    temperature: float | None, top_p: float | None, top_k: int | None
) -> dict:
    """Build mlx-lm's ``sampler`` kwarg from per-model overrides, or {} when none are set (so
    the library keeps its own greedy/default sampling). Isolated so the mlx-lm import stays
    lazy and a sampler-API change touches only here."""
    if temperature is None and top_p is None and top_k is None:
        return {}
    from mlx_lm.sample_utils import make_sampler

    return {
        "sampler": make_sampler(
            temp=float(temperature) if temperature is not None else 0.0,
            top_p=float(top_p) if top_p is not None else 0.0,
            top_k=int(top_k) if top_k is not None else 0,
        )
    }


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared leading run of two token sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _default_trimmer(prompt_cache: object, drop: int) -> int:
    """Trim ``drop`` tokens off a banked cache; returns how many actually came off
    (``trim_prompt_cache`` is all-or-nothing across layers, so != drop means "don't reuse")."""
    from mlx_lm.models.cache import trim_prompt_cache

    return trim_prompt_cache(prompt_cache, drop)


class PromptCacheStore:
    """Small LRU bank of finished turns' KV caches, keyed by the token ids they represent.

    Requests that interleave between conversations would otherwise evict each other every
    turn, leaving only the shared system prefix reusable — measured on four concurrent coding
    sessions as ``cached`` pinned at 4979 while prefill ran 20-50k tokens, 17-67s, per step.
    ``take`` removes the entry while its cache is in flight (a KV cache is mutable — two
    requests must never share one), and the finished request banks the extended cache back.

    ``max_tokens`` bounds the bank by the thing that actually costs: a KV cache is a fixed
    number of bytes per token for a given model, so a token budget IS a byte budget. Left
    None the bank is bounded only by ``max_entries``, which is fine where turns are short and
    ruinous where they are not — an agent turn reaching 120k tokens is ~12GB of KV on a
    30B model, and four of those do not fit anywhere.
    """

    def __init__(self, max_entries: int = 4, trimmer=None, max_tokens: int | None = None):
        self._entries: list[tuple[list[int], object]] = []
        self._max = max_entries
        self._max_tokens = max_tokens
        self._trim = trimmer or _default_trimmer

    def take(self, full_ids: list[int]) -> tuple[list[int], object | None]:
        """(suffix to prefill, cache to resume from) — or (full_ids, None) for a cold start."""
        best_i, best_common = -1, 0
        for i, (ids, _cache) in enumerate(self._entries):
            common = _common_prefix_len(ids, full_ids)
            if common > best_common:
                best_i, best_common = i, common
        best_common = min(best_common, len(full_ids) - 1)  # always leave ≥1 token to feed
        if best_i < 0 or best_common <= 0:
            return full_ids, None
        ids, _ = self._entries[best_i]
        drop = len(ids) - best_common
        # Reuse TRIMS the entry back to the shared prefix, destroying the rest — which belongs to
        # whichever conversation banked it. That's the right trade when this turn continues that
        # conversation (the tail dropped is a couple of template tokens) and the wrong one when it
        # merely shares the system prompt: we'd save `best_common` tokens now and cost that
        # conversation `drop` on its next turn. Without this guard the bank never holds more than
        # one entry at all — serial generation means every turn takes the only cache there is,
        # trims it to the system prefix and banks it back, which is the single-slot behaviour this
        # bank replaced. So while a slot is free, only cannibalize an entry we keep more of than we
        # drop. Once full there is nothing left to protect: every choice costs an entry, so take
        # the longest match.
        if drop > best_common and len(self._entries) < self._max:
            return full_ids, None
        _, cache = self._entries.pop(best_i)
        if drop and self._trim(cache, drop) != drop:
            # Untrimmable (e.g. a wrapped rotating layer) — a cache out of step with its ids
            # would corrupt the turn, so pay the full prefill instead.
            return full_ids, None
        return full_ids[best_common:], cache

    def put(self, ids: list[int], cache: object) -> None:
        self._entries.append((list(ids), cache))
        while len(self._entries) > self._max:
            self._entries.pop(0)
        # Never evict what was just banked: one conversation longer than the whole budget still
        # gets its cache back, because dropping it would mean re-prefilling it every single turn
        # — the exact cost this bank exists to avoid.
        while (
            self._max_tokens is not None
            and len(self._entries) > 1
            and sum(len(ids) for ids, _ in self._entries) > self._max_tokens
        ):
            self._entries.pop(0)


# How many conversations one engine keeps warm, and the ceiling on what they may hold.
#
# A KV cache costs a fixed number of bytes per token for a given model, so a token budget IS a
# byte budget: Qwen3-Coder-30B-A3B is 48 layers x 4 KV heads x 128 head_dim x 2 (K+V) x 2 bytes
# = 96KB/token, making this budget ~24GB. Sized against what the bank is actually asked to
# hold: an agent turn runs 60-90k tokens, so anything under ~200k fits one such conversation
# and evicts the second — which is the single-slot behaviour this replaced, just later.
# Deliberately a flat number rather than a fraction of RAM: it is only honest for a model of
# this size, and the pool's own admission ceiling (which books each engine's measured working
# memory, this bank included) is what stops a load that would not fit.
# Four slots because that is what a person actually drives at once (a few chats, a few agent
# tracks); the budget, not the slot count, is what keeps it honest.
_CACHE_SLOTS = 4
_CACHE_TOKEN_BUDGET = 256_000


class MlxEngine:
    """A loaded mlx-lm model + tokenizer that streams text for one chat turn."""

    def __init__(self, model: object, tokenizer: object, harmony: bool = False):
        self._model = model
        self._tokenizer = tokenizer
        # Harmony (gpt-oss) checkpoints need their reasoning text mapped back into the
        # template's thinking/content fields at render time (N82/N84). Decided once at
        # load (vocab carries <|call|>), so render never guesses per turn.
        self._harmony = harmony
        # Prompt caches reused across turns (see _prefill_plan). Safe without a lock: the pool
        # pins each model to a single-thread executor, so all of a model's generation is
        # serialized — but *sequential* is not *single-conversation*. One slot assumed the next
        # request continues the last one, which a single chat does and several do not: with four
        # sessions in flight the previous generation was always somebody else's, so every turn
        # reused only the shared system prefix and re-prefilled tens of thousands of tokens.
        self._caches = PromptCacheStore(
            max_entries=_CACHE_SLOTS, max_tokens=_CACHE_TOKEN_BUDGET
        )
        # High-water working memory (KV cache + activations) this model needed for a turn, over and
        # above its resident weights. The pool harvests it at admission time so a load is budgeted
        # against what generation actually costs rather than a guess (N86).
        self.working_memory_bytes = 0

    def _encode_for_generation(self, prompt: str) -> list[int]:
        # Mirror stream_generate's own tokenisation exactly so the ids we prefix-match against the
        # cache are the ids the model actually sees: BOS is added only when the prompt doesn't
        # already start with it.
        bos = getattr(self._tokenizer, "bos_token", None)
        add_special = bos is None or not prompt.startswith(bos)
        return list(self._tokenizer.encode(prompt, add_special_tokens=add_special))

    def _new_prompt_cache(self) -> object:
        """Build a prompt cache every layer of which can be trimmed, so prefix reuse actually works.

        mlx-lm gives each sliding-window layer a ``RotatingKVCache``, whose ``is_trimmable()`` is
        False once its ring has wrapped (``offset >= max_size``, i.e. after ~128 tokens). Since
        ``trim_prompt_cache`` is all-or-nothing, ONE such layer makes the whole cache untrimmable
        and every turn re-prefills the entire prompt: measured on gpt-oss-120b as 45 consecutive
        generations at ``cached=0``, prompts growing to 13.7k and prefill to 12s per step. This is
        not recoverable by trimming smarter — a wrapped ring has already overwritten the keys a
        rolled-back window would need, which is exactly why mlx-lm reports it untrimmable.

        A plain ``KVCache`` is equivalent for these layers because the model masks the window
        itself: sliding layers always call ``create_attention_mask(..., window_size=W)``, and
        ``cache.create_attention_mask`` tests ``window_size is not None`` *before* the ``N == 1``
        shortcut, so both prefill and decode get a banded mask. Verified against mlx-lm 0.31.3 at
        offset 500: rotating and full caches each attend exactly 128 keys, for N=1 and N=3. RoPE is
        unaffected — it reads ``cache.offset``, which both classes track absolutely.

        The cost is memory: sliding layers keep the whole sequence instead of ``max_size`` tokens.
        For gpt-oss (half its layers sliding) that is 2x the KV total — +500MB at a 13.7k context —
        against re-prefilling 13.7k tokens through 36 layers on *every* step, which is what the
        machine was actually dying of.
        """
        from mlx_lm.models import cache as cache_mod

        cache = cache_mod.make_prompt_cache(self._model)
        # getattr, not import: the fake mlx_lm in tests defines only the two functions, and a
        # future mlx-lm that drops either name should degrade to today's behaviour, not crash.
        rotating = getattr(cache_mod, "RotatingKVCache", None)
        kv_cls = getattr(cache_mod, "KVCache", None)
        if rotating is None or kv_cls is None:
            return cache
        return [kv_cls() if isinstance(entry, rotating) else entry for entry in cache]

    def _prefill_plan(self, full_ids: list[int]) -> tuple[list[int], object]:
        """Reuse the KV of the banked turn sharing the longest prefix with this one, and prefill
        only the new tail. Claude Code re-sends the whole conversation every turn, so without this
        each turn re-prefills tens of thousands of tokens (system prompt + tools + history + file
        reads) before the first output token — the dominant per-turn cost. Falls back to a fresh
        full prefill when nothing is shared or no cache can be trimmed back to the shared prefix.
        """
        suffix, cache = self._caches.take(full_ids)
        return suffix, cache if cache is not None else self._new_prompt_cache()

    def encode_prompt(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> list[int]:
        """Rendered prompt as generation-ready token ids — the single place template rendering
        and BOS handling meet, shared by the serial path, token counting and the batch lane
        (``mlx_batch``), so all three see byte-identical prompts."""
        prompt = _render_prompt(
            self._tokenizer.apply_chat_template, messages, tools, chat_template_kwargs,
            harmony=self._harmony,
        )
        return self._encode_for_generation(prompt)

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> int:
        """Token length of the rendered prompt — powers /v1/messages/count_tokens and the usage
        Claude Code reads to track how full the context is."""
        return len(self.encode_prompt(messages, tools, chat_template_kwargs))

    def stream_text(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        chat_template_kwargs: dict | None = None,
        usage_out: dict | None = None,
        **_ignored,
    ) -> Iterator[str]:
        # Imported lazily: MLX is a heavy, Apple-Silicon-only dependency.
        from mlx_lm import stream_generate

        # Passing tools lets the chat template render them in the model's expected
        # tool-calling format. _render_prompt (via encode_prompt) normalises tool_calls and
        # handles the tools-kwarg fallback without masking real template errors.
        full_ids = self.encode_prompt(messages, tools, chat_template_kwargs)
        if usage_out is not None:
            # Full prompt count regardless of cache reuse — it's the context size Claude Code tracks.
            usage_out["input_tokens"] = len(full_ids)

        suffix, cache = self._prefill_plan(full_ids)
        generated: list[int] = []
        committed = False
        last = None
        t0 = time.monotonic()
        ttft: float | None = None
        # Baseline for the working-memory measurement below: everything MLX holds before this turn
        # allocates anything (i.e. the resident weights, this model's and any other's).
        resident_before = _active_memory_bytes()
        _reset_peak_memory()
        try:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                suffix,
                max_tokens=max_tokens,
                prompt_cache=cache,
                **_sampler_kwargs(temperature, top_p, top_k),
            ):
                if ttft is None:
                    ttft = time.monotonic() - t0  # ≈ prefill time (first token out)
                last = response
                token = getattr(response, "token", None)
                if token is not None:
                    generated.append(token)
                text = getattr(response, "text", None)
                if text:
                    yield text
            committed = True
            # The line that settles "why is this slow": prefill (cache miss cost, fixable by
            # cache/session hygiene) vs decode (the model's ceiling, not fixable). cached = tokens
            # served from the reused KV; prefill = tokens actually recomputed this turn.
            log.info(
                "generation: prompt=%d (cached=%d, prefill=%d) prefill=%.2fs decode=%d tok "
                "in %.2fs (%.1f tok/s)",
                len(full_ids), len(full_ids) - len(suffix), len(suffix), ttft or 0.0,
                len(generated), max(time.monotonic() - t0 - (ttft or 0.0), 0.0),
                getattr(last, "generation_tps", 0.0) or 0.0,
            )
        finally:
            # What this turn cost beyond the resident weights. Measured even on an early stop —
            # a turn killed by a disconnect still peaked. Both readings are process-wide, so a
            # second model generating concurrently can clip this one's peak; keeping the
            # high-water mark across turns lets a later uncontended turn correct it upward.
            peak = _peak_memory_bytes()
            if peak is not None and resident_before is not None:
                self.working_memory_bytes = max(
                    self.working_memory_bytes, peak - resident_before
                )
            if committed:
                # The cache now holds the whole prompt plus everything generated; bank that so the
                # next turn of this conversation reuses it instead of re-prefilling from scratch.
                self._caches.put(full_ids + generated, cache)
                if usage_out is not None:
                    usage_out["output_tokens"] = len(generated)
            # An early stop / error / client disconnect leaves the cache extended past the ids it
            # was taken under, so it is simply not banked — ``take`` already removed it, and a
            # mismatched cache would corrupt the next turn's prefix reuse.


# APC pool geometry. Blocks are the unit of KV reuse; a lookup matches whole blocks of a
# prompt's leading tokens, so a smaller block wastes less at the boundary and 16 is the
# library's own default.
#
# The block count is the only real decision, and it is a memory budget: this model's KV is
# 2 (K and V) × 64 layers × 4 KV heads × 256 head_dim × 2 bytes = 256 KB per token, so 4096
# blocks is 65,536 tokens ≈ 16 GB held beside ~30 GB of weights. That is sized for what this
# backend is actually asked to do — an agent turn whose conversation grows to 60–90k tokens
# across forty tool calls — where the pool has to hold one turn's prefix for the next
# iteration to match it. Raise it on a machine with room; the cost is linear and paid in
# resident memory, the benefit is prefill that stops growing with the conversation.
_APC_BLOCK_SIZE = 16
_APC_NUM_BLOCKS = 4096

# How many exact-prefix snapshots the pool keeps, and it is a concurrency budget rather than a
# memory one. A hybrid-attention model cannot reuse block by block — its recurrent state is not
# block-concatenable — so every reuse on this path goes through APC's *exact* cache, and every
# generation stores one snapshot into it. mlx-vlm's own default is 2, which is enough for one
# conversation and exactly wrong for two: with A and B alternating, A's store evicts B's
# snapshot and B's store evicts A's, so every single lookup misses.
#
# Not theoretical. Measured 2026-08-20 driving two agent tracks against this backend: `apc
# reused=0` on all nine interleaved generations with prefill at 8–30 s, against `reused=8711`
# and prefill 0.8–5 s the moment one track was left running alone. Wall clock per track was
# about six times worse, which made two concurrent tracks slower in total than running them one
# after the other.
#
# Six because it is two snapshots each for the three tracks a person actually drives at once,
# and one entry is a full KV snapshot of that conversation's prompt — at 256 KB/token an
# agent turn's 10k-token prompt is ~2.5 GB, so this is the number to lower on a smaller
# machine. Set through the environment because the library reads it in `APCManager.__init__`
# and takes no argument for it; `setdefault`, so an operator who has chosen a number keeps it.
_APC_EXACT_CACHE_ENTRIES = "6"


def _apc_manager_for(model: object):
    """A prefix cache for ``model``, or None where one cannot be used.

    One manager per loaded model, never shared: APC keys blocks by the hash of the token ids
    alone, with nothing in it that says which model produced them, so a pool serving two
    models would hand one of them the other's K/V.

    Off rather than fatal when the model's cache layout isn't one APC can reconstruct — the
    alternative is an AttributeError from inside ``stream_generate`` on every turn — but the
    reason is logged, because "the assistant got slower" is otherwise the only symptom.
    """
    try:
        from mlx_vlm import apc as vlm_apc
    except Exception as exc:  # an mlx-vlm too old to have APC at all
        log.info("prefix cache unavailable (mlx-vlm has no apc module): %s", exc)
        return None
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        log.info("prefix cache off: this model exposes no language_model to cache")
        return None
    mode = vlm_apc.model_apc_mode(language_model)
    if mode is None:
        log.info("prefix cache off: this model's cache layout is not one APC can rebuild")
        return None
    os.environ.setdefault("APC_EXACT_CACHE_ENTRIES", _APC_EXACT_CACHE_ENTRIES)
    log.info(
        "prefix cache on (mode=%s, %d blocks × %d tokens, %s exact snapshots)",
        mode, _APC_NUM_BLOCKS, _APC_BLOCK_SIZE,
        os.environ["APC_EXACT_CACHE_ENTRIES"],
    )
    return vlm_apc.APCManager(num_blocks=_APC_NUM_BLOCKS, block_size=_APC_BLOCK_SIZE)


class VlmChatEngine:
    """A loaded mlx-vlm model used as a *text* chat model.

    Omni / vision-language checkpoints (e.g. Qwen-VL) carry a ``vision_config`` and
    mlx-lm can't load them — only mlx-vlm can. This wraps mlx-vlm so such a model can
    still be the chat model (text in, text out), mirroring ``MlxEngine``'s contract.
    Image *input* isn't wired into chat here; the agent reads images via the separate
    ``view_image`` tool. The processor proxies the model's chat template, so multi-turn
    role handling matches the LLM path.

    Prompt reuse is APC's rather than ``MlxEngine``'s ``PromptCacheStore``: mlx-vlm takes no
    ``prompt_cache`` from a caller, but it does take a prefix-cache manager and match a new
    prompt's leading blocks against it. Without one, every tool call in an agent turn
    re-prefills the whole conversation from the system prompt down, which is quadratic in the
    number of calls and was the dominant cost of a long turn — minutes per call by the
    fortieth, against seconds for the first.
    """

    def __init__(self, model: object, processor: object):
        self._model = model
        self._processor = processor
        self._apc = _apc_manager_for(model)
        # Matched-token total as of the last generation, so each line can report its own turn.
        self._apc_matched = 0

    def _generate_kwargs(self) -> dict:
        """What this engine adds to ``stream_generate`` beyond the prompt. Empty here; the
        speculative subclass is the reason it exists, so that both share one generate loop
        and one place where prefix caching and the timing line are wired."""
        return {}

    def _templater(self):
        return getattr(self._processor, "apply_chat_template", None) or getattr(
            self._processor, "tokenizer"
        ).apply_chat_template

    def count_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> int:
        """Token length of the rendered prompt, via the processor's tokenizer. Mirrors MlxEngine so
        /v1/messages/count_tokens works for VLM-loaded chat models too."""
        prompt = _render_prompt(self._templater(), messages, tools, chat_template_kwargs)
        tok = getattr(self._processor, "tokenizer", None) or self._processor
        return len(tok.encode(prompt))

    def stream_text(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        chat_template_kwargs: dict | None = None,
        usage_out: dict | None = None,
        **_ignored,
    ) -> Iterator[str]:
        from mlx_vlm import stream_generate

        prompt = _render_prompt(self._templater(), messages, tools, chat_template_kwargs)
        kwargs = dict(self._generate_kwargs())
        if self._apc is not None:
            # No tenant: the namespace only exists to keep conversations apart, and keeping
            # them together is the point — every turn of every session shares a system prompt,
            # and that prefix is the largest block run any of them can match.
            kwargs["apc_manager"] = self._apc
        t0 = time.monotonic()
        ttft: float | None = None
        last = None
        for chunk in stream_generate(
            self._model, self._processor, prompt, max_tokens=max_tokens, **kwargs
        ):
            # Draft chunks are the drafter's speculative preview; the accepted text arrives
            # again in normal chunks — forwarding both would duplicate output. Always checked,
            # not only under the speculative subclass: a plain run has no draft chunks to skip.
            if getattr(chunk, "is_draft", False):
                continue
            if ttft is None:
                ttft = time.monotonic() - t0  # ≈ prefill time (first token out)
            last = chunk
            text = getattr(chunk, "text", None)
            if text is None and isinstance(chunk, str):
                text = chunk
            if text:
                yield text
        if usage_out is not None and last is not None:
            usage_out["input_tokens"] = getattr(last, "prompt_tokens", 0) or 0
            usage_out["output_tokens"] = getattr(last, "generation_tokens", 0) or 0
            # "length" means the reply stopped because it ran out of budget, not because the
            # model finished. Reported so the agent loop can say so: a truncated reply with no
            # tool call in it is indistinguishable from a considered answer, and gets acted on
            # as one — which is how a turn that ran out mid-sentence reports success.
            usage_out["finish_reason"] = getattr(last, "finish_reason", None)
        self._log_generation(last, ttft, t0)

    def _log_generation(self, last: object, ttft: float | None, t0: float) -> None:
        """The line that settles "why is this slow" on this path, as ``MlxEngine`` logs it on
        the other: prefill is what a cache miss costs and is fixable by cache hygiene, decode
        is the model's ceiling and is not.

        ``reused`` is this turn's own matched-token count, differenced from the manager's
        running totals rather than read off its ``token_hit_rate``. That ratio is served
        tokens against matched ones, and the exact-snapshot path — the only one a hybrid
        attention model can use — never counts a served token, so the ratio reads 100% for a
        turn that reused sixteen tokens of a twenty-thousand-token prompt. A number that is
        always 100% is worse than no number: it answers the question this line exists to ask.
        """
        if last is None:
            return
        reused = 0
        if self._apc is not None:
            stats = self._apc.stats_snapshot()
            total = stats.get("matched_tokens", 0)
            reused, self._apc_matched = total - self._apc_matched, total
        log.info(
            "generation: prompt=%d prefill=%.2fs decode=%d tok (%.1f tok/s) apc reused=%d",
            getattr(last, "prompt_tokens", 0) or 0,
            ttft or 0.0,
            getattr(last, "generation_tokens", 0) or 0,
            getattr(last, "generation_tps", 0.0) or 0.0,
            reused,
        )


class SpeculativeVlmEngine(VlmChatEngine):
    """A VlmChatEngine that decodes with a speculative drafter (MTP/DFlash/EAGLE).

    MTP heads are split off their target model (e.g. ``…-27B-MTP-8bit``) and need the target's
    hidden states each step, so only mlx-vlm's speculative loop can drive them — mlx-lm's
    ``draft_model`` kwarg expects a standalone LM and cannot. That is why a drafted model loads
    through mlx-vlm even when mlx-lm could serve it plainly; the trade is mlx-lm's own prompt
    cache for a faster decode, and APC on the base class is what buys the reuse back.
    """

    def __init__(self, model: object, processor: object, draft_model: object, draft_kind: str):
        super().__init__(model, processor)
        self._draft_model = draft_model
        self._draft_kind = draft_kind

    def _generate_kwargs(self) -> dict:
        return {"draft_model": self._draft_model, "draft_kind": self._draft_kind}


def _load_speculative(path: Path, draft_path: Path) -> SpeculativeVlmEngine:
    """Load a chat model paired with a speculative drafter (the per-model "draft" setting).

    The target loads through mlx-vlm (``_load_vlm``: same engine the speculative loop needs,
    chat-template refusal included); the drafter loads via mlx-vlm's own ``load_drafter``, which
    resolves the draft kind (mtp/dflash/eagle3) from the drafter's model_type. Compatibility is
    validated structurally (hidden sizes), so a drafter for the wrong target fails loud here
    instead of generating garbage.
    """
    engine = _load_vlm(path)
    try:
        from mlx_vlm.speculative.drafters import (
            load_drafter,
            validate_drafter_compatibility,
        )
    except ImportError as exc:  # an mlx-vlm too old to know drafters
        raise RuntimeError(
            "speculative decoding needs mlx-vlm with drafter support — update mlx-vlm "
            "in Settings ▸ Managed tools, or clear this model's draft setting."
        ) from exc
    draft_model, draft_kind = load_drafter(str(draft_path))
    validate_drafter_compatibility(engine._model, draft_model, draft_kind)
    log.info("loaded speculative drafter (%s): %s", draft_kind, draft_path)
    return SpeculativeVlmEngine(engine._model, engine._processor, draft_model, draft_kind)


def _load_llm(path: Path) -> MlxEngine:
    from mlx_lm import load

    try:
        model, tokenizer = load(str(path))
    except Exception as exc:
        # mlx-lm raises a bare ``ValueError("Model type <x> not supported.")`` when the
        # installed mlx-lm predates a model's architecture (e.g. qwen3_5 / qwen3_6). The
        # checkpoint is fine — the venv's mlx-lm is just too old. Re-raise with the version
        # and a pointer to the in-app updater instead of a cryptic message (N11).
        msg = str(exc)
        if "not supported" in msg.lower() and "model type" in msg.lower():
            from importlib import metadata

            try:
                installed = metadata.version("mlx-lm")
            except Exception:
                installed = "unknown"
            raise RuntimeError(
                f"{msg.rstrip('.')}. Installed mlx-lm is {installed}, too old for this "
                "model's architecture — update it in Settings ▸ Managed tools (更新套件), "
                "then restart the backend."
            ) from exc
        raise
    # Harmony (gpt-oss) hands control back to the runtime at <|call|>, but neither the
    # checkpoint's generation_config (eos = <|return|>/<|endoftext|> only) nor mlx-lm
    # registers it as a stop token — generation blew straight past the first tool call
    # and the model hallucinated results + more calls until the loop's thrash guard
    # killed the turn (N83). Register it whenever the vocab carries the exact token;
    # the round-trip check keeps an unk-mapping tokenizer from adding unk as EOS.
    harmony = False
    try:
        call_id = tokenizer.convert_tokens_to_ids("<|call|>")
        if call_id is not None and tokenizer.convert_ids_to_tokens(call_id) == "<|call|>":
            tokenizer.add_eos_token("<|call|>")
            harmony = True
    except Exception:  # vocab probing must never break a load
        pass
    return MlxEngine(model, tokenizer, harmony=harmony)


def _load_vlm(path: Path) -> VlmChatEngine:
    try:
        from mlx_vlm import load
    except ImportError as exc:  # vlm models need the optional extra
        raise RuntimeError(
            'this is a vision-language model; install mlx-vlm to chat with it: '
            'uv pip install -e ".[vlm]"'
        ) from exc
    try:
        model, processor = load(str(path))
    except Exception as exc:
        # mlx-vlm loads weights strictly: a checkpoint whose architecture doesn't match
        # mlx-vlm's class for its model_type raises a ValueError that lists EVERY mismatched
        # weight — hundreds of names (e.g. Lance-3B-Video, a custom "omni" understand+generate
        # variant declaring model_type qwen2_5_vl but carrying extra gen heads/VAE bridges the
        # stock class lacks). Collapse it to a one-line, actionable error so the gateway/GUI
        # shows a usable message instead of a multi-KB key dump; the full detail stays in the
        # backend log via the caller's log.exception.
        head = next((ln.strip() for ln in str(exc).splitlines() if ln.strip()), "") or type(
            exc
        ).__name__
        raise RuntimeError(
            f"this vision-language model failed to load — its checkpoint doesn't match "
            f"mlx-vlm's architecture ({head}). It's likely a custom or unsupported variant; "
            f"pick a standard chat model instead."
        ) from exc
    # A checkpoint that ships no chat template can't be chatted with: mlx-vlm's
    # apply_chat_template swallows the ``tools`` kwarg and silently degrades to bare
    # "System:/User:/Assistant:" text — no turn tokens, no tool schemas, no stop
    # discipline — so the model roleplays both sides for tens of thousands of tokens
    # per turn (N80, gemma-4-12B-bf16). Refuse at load so the user gets one clear
    # error instead of a silent 50-minute garbage turn.
    template = getattr(processor, "chat_template", None) or getattr(
        getattr(processor, "tokenizer", None), "chat_template", None
    )
    if not template:
        raise RuntimeError(
            "this model's checkpoint ships no chat template (likely an incomplete "
            "conversion) — chat would silently render as untemplated text and the "
            "model rambles instead of answering. Re-download it or pick another "
            "checkpoint."
        )
    return VlmChatEngine(model, processor)


def _default_loader(
    path: Path, forced_kind: str | None = None, draft_path: Path | None = None
) -> object:
    # Dispatch by model family: a VL/omni checkpoint loads through mlx-vlm, everything else
    # through mlx-lm. ``forced_kind`` (a per-model type override) wins over auto-detection — it is
    # how a checkpoint we'd misclassify as VLM, and then crash the mlx-vlm loader on, is told to
    # load as a plain LLM and just work. When unset, classify_kind re-reads config.json from path.
    # A paired drafter (per-model "draft" setting) forces the mlx-vlm speculative path — the only
    # loop that can drive an MTP head — regardless of which engine would serve the model plainly.
    if draft_path is not None:
        return _load_speculative(path, draft_path)
    from .mlx_discovery import classify_kind

    kind = forced_kind or classify_kind(path)
    return _load_vlm(path) if kind == "vlm" else _load_llm(path)


class MlxEnginePool:
    """LRU pool of loaded engines. ``loader`` is injectable for tests."""

    def __init__(
        self,
        *,
        max_loaded: int | None = 1,
        loader: Callable[..., object] | None = None,
        pinned: set[str] | None = None,
        mem_ceiling_bytes: int | None = None,
    ):
        # None / <=0 means "no count cap — the memory ceiling alone gates residency", which is
        # what lets several models stay resident when they fit (fusion prefetch, fast switching).
        self._max = max_loaded if (max_loaded and max_loaded > 0) else None
        self._loader = loader or _default_loader
        # Insertion order == LRU order; move_to_end marks most-recently-used.
        self._loaded: OrderedDict[str, object] = OrderedDict()
        # model_id -> bytes its WEIGHTS hold (measured active-memory delta, or disk estimate as a
        # fallback). Drives byte-level admission against the ceiling, alongside the count cap.
        self._footprint: dict[str, int] = {}
        # model_id -> peak working memory (KV cache + activations) measured while it generated.
        # Survives eviction on purpose: it's learned knowledge about the model, not about this
        # residency, so re-admitting it later budgets the real number instead of guessing again.
        self._working_memory: dict[str, int] = {}
        # model_id -> its dedicated single-thread executor. MLX (mlx-vlm especially) binds a GPU
        # stream to the thread a model was LOADED on; generating from any other thread raises
        # "There is no Stream(gpu, N) in current thread". So every engine gets one thread for its
        # whole life, and both the load and all generation run there (thread affinity).
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._pinned: set[str] = set(pinned or set())
        # None / <=0 disables byte-admission entirely — the pool then behaves exactly as the
        # count-only LRU it was before, so the guardrail is strictly opt-in via mem_ceiling_gb.
        self._ceiling = mem_ceiling_bytes if (mem_ceiling_bytes and mem_ceiling_bytes > 0) else None
        self._lock = asyncio.Lock()

    def _effective_max(self) -> int | None:
        """Count cap actually enforced: the configured one, or — when neither a cap nor a memory
        ceiling exists — a failsafe of 1 so an unbounded pool can't OOM a machine whose RAM we
        couldn't detect. None means "no count cap" (ceiling-gated)."""
        if self._max is not None:
            return self._max
        return None if self._ceiling is not None else 1

    async def acquire(
        self,
        model_id: str,
        path: Path,
        forced_kind: str | None = None,
        draft_path: Path | None = None,
    ) -> object:
        """Return the loaded engine for ``model_id``, loading (and evicting) as needed.
        ``forced_kind`` overrides the loader's auto-detection (per-model type override);
        ``draft_path`` pairs a speculative drafter checkpoint with the load. Both matter only
        on a cold load — changing them for a resident model takes effect after unload.

        Raises ``ModelAdmissionError`` *before* loading if a memory ceiling is set and the model
        can't be made to fit — fail loud rather than OOM-crash."""
        # Lock-free fast path: pool state only mutates from the event loop and this path has no
        # awaits, so it's atomic w.r.t. other coroutines. Without it, a resident model's acquire
        # would queue behind whatever load currently holds the lock — i.e. generation on model A
        # would stall for the full duration of model B's background prefetch.
        engine = self._loaded.get(model_id)
        if engine is not None:
            self._loaded.move_to_end(model_id)
            return engine
        async with self._lock:
            if model_id in self._loaded:  # loaded while we waited (e.g. by its own prefetch)
                self._loaded.move_to_end(model_id)
                return self._loaded[model_id]
            incoming = _estimate_model_bytes(path)
            self._harvest_working_memory()
            self._make_room(exclude=model_id, incoming=incoming)
            # Load outside would race the lock; loading under the lock serialises
            # model switches, which is what we want (one heavy load at a time).
            log.info("loading model into pool: %s (~%.1fGB est)", model_id, incoming / 1e9)
            before = _active_memory_bytes()
            # Load on the model's OWN thread — generation must later run on this same thread
            # (mlx-vlm stream affinity), so the executor is created first and the load is its
            # first job. Torn down when the model leaves the pool.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mlx-{model_id}")
            # draft_path is appended only when set so injected test loaders (and any older
            # two-arg loader) keep their (path, forced_kind) signature.
            args = (path, forced_kind) if draft_path is None else (path, forced_kind, draft_path)
            try:
                engine = await asyncio.get_running_loop().run_in_executor(
                    executor, self._loader, *args
                )
            except BaseException:
                # A failed load still ran mlx code on this thread, so it retires like any other.
                _retire_executor(executor)
                raise
            self._executors[model_id] = executor
            self._loaded[model_id] = engine
            self._loaded.move_to_end(model_id)
            # Refine the disk estimate with the real footprint when MLX exposes it. The lock is
            # held and the cache was just cleared on eviction, so the active-memory delta is this
            # model's own contribution; fall back to the estimate (tests / non-MLX / no delta).
            after = _active_memory_bytes()
            measured = (after - before) if (before is not None and after is not None and after > before) else 0
            self._footprint[model_id] = measured or incoming
            log.info(
                "model loaded: %s (%.1fGB weights, pool now: %s = %.1fGB budgeted%s)",
                model_id, self._footprint[model_id] / 1e9, self.loaded_ids(),
                self._loaded_bytes() / 1e9,
                f"/{self._ceiling / 1e9:.0f}GB ceiling" if self._ceiling else "",
            )
            return engine

    def admission_bytes(self, model_id: str, weights: int) -> int:
        """What admitting ``model_id`` costs the budget: its weights plus the working memory it
        needs to generate — measured if this pool has watched it, else the cold-start factor.
        Public so the fusion scheduler can size prefetches with the exact policy the gate
        enforces; when the two disagree the scheduler under-books every load by the reserve."""
        measured = self._working_memory.get(model_id)
        if measured is not None:
            return weights + measured
        return int(weights * _WORKING_MEMORY_FACTOR)

    def _admission_cost(self, model_id: str) -> int:
        """What a resident model really costs the memory budget: its weights plus the working
        memory generation needs. Uses the measured working memory once the model has streamed a
        turn, and ``_WORKING_MEMORY_FACTOR`` as the cold-start guess until then."""
        weights = self._footprint.get(model_id, 0)
        measured = self._working_memory.get(model_id)
        if measured is not None:
            return weights + measured
        return int(weights * _WORKING_MEMORY_FACTOR)

    def _loaded_bytes(self, exclude: str | None = None) -> int:
        """Total budget the resident models hold — admission costs, not bare weights, so the
        number compared against the ceiling is the one that has to survive generation."""
        return sum(self._admission_cost(k) for k in self._loaded if k != exclude)

    def _drop(self, model_id: str, reason: str) -> None:
        self._loaded.pop(model_id, None)
        self._footprint.pop(model_id, None)
        # Non-blocking: an in-flight generation on this thread finishes on its own (the worker holds
        # the engine reference); the executor just stops accepting new work, after one final
        # compile-cache clear. That clear is not optional — see ``_retire_executor``; the caller
        # (``_make_room``) runs ``_release_mlx_memory`` microseconds later, and a thread exiting into
        # MLX's teardown during that GC is what segfaulted the backend.
        executor = self._executors.pop(model_id, None)
        if executor is not None:
            _retire_executor(executor)
        log.info("evicted model from pool: %s (%s)", model_id, reason)

    def _make_room(self, exclude: str, incoming: int) -> None:
        """Evict LRU non-pinned engines until both the count cap (``max_loaded``) and, when a
        ceiling is set, the memory budget (held + ``incoming`` <= ceiling) are satisfied. Dropping
        the last reference is necessary but not sufficient to free MLX's unified memory —
        ``_release_mlx_memory`` (once, after any eviction) clears the Metal buffer cache too.

        Raises ``ModelAdmissionError`` if, after evicting everything evictable, the incoming model
        still overflows the ceiling (it's larger than the budget, or pinned models leave no room)."""
        evicted = False
        # The incoming model needs room to generate too, not just to sit there.
        incoming = self.admission_bytes(exclude, incoming)

        def next_victim() -> str | None:
            return next((k for k in self._loaded if k != exclude and k not in self._pinned), None)

        # Count cap: make room for one more (ceiling-independent — preserves prior behaviour).
        # No cap (None) → residency is governed by the memory ceiling alone.
        cap = self._effective_max()
        while cap is not None and len(self._loaded) >= cap:
            victim = next_victim()
            if victim is None:
                break  # everything left is pinned — exceed the count budget rather than evict it
            self._drop(victim, f"making room for {exclude}")
            evicted = True

        # Memory ceiling: keep evicting until the incoming estimate fits under the budget.
        if self._ceiling is not None:
            while self._loaded_bytes(exclude) + incoming > self._ceiling:
                victim = next_victim()
                if victim is None:
                    break
                self._drop(victim, f"freeing memory for {exclude}")
                evicted = True

        if evicted:
            _release_mlx_memory()

        if self._ceiling is None:
            return
        # The books say what SHOULD be resident; MLX says what IS. They diverge when an evicted
        # model's memory hasn't come back yet — ``_drop`` can't wait for an in-flight generation
        # to release its engine (a long turn would block every model switch), so the books can
        # read "free" while those weights are still physically there. Budgeting on the books
        # alone stacks the incoming model on memory that never left: a fusion panel evicted a
        # 70GB judge and loaded 79GB one second later, and the OS killed the backend.
        booked = self._loaded_bytes(exclude)
        actual = _active_memory_bytes() or 0
        held = max(booked, actual)
        if held + incoming <= self._ceiling:
            return
        stale = actual > booked  # memory from an evicted model hasn't been reclaimed yet
        raise ModelAdmissionError(
            f"{exclude} needs ~{incoming / 1e9:.1f}GB but only "
            f"{max(0.0, (self._ceiling - held) / 1e9):.1f}GB is free under the "
            f"{self._ceiling / 1e9:.0f}GB memory limit"
            + (
                f" ({actual / 1e9:.1f}GB is still held by a model that was just evicted — a "
                "generation still running on it hasn't released its weights; retry once it finishes)"
                if stale
                else f" ({held / 1e9:.1f}GB held by pinned models)" if held else ""
            )
            + ". Free memory (unload other models / close apps) or pick a smaller model."
        )

    async def load(
        self,
        model_id: str,
        path: Path,
        forced_kind: str | None = None,
        draft_path: Path | None = None,
    ) -> None:
        await self.acquire(model_id, path, forced_kind, draft_path)

    async def unload(self, model_id: str) -> bool:
        async with self._lock:
            removed = self._loaded.pop(model_id, None) is not None
            if removed:
                self._footprint.pop(model_id, None)
                executor = self._executors.pop(model_id, None)
                if executor is not None:
                    _retire_executor(executor)  # never a bare shutdown — see _retire_executor
                log.info("unloading model from pool: %s (pool now: %s)", model_id, self.loaded_ids())
                _release_mlx_memory()
            else:
                log.info("unload requested for model not in pool: %s", model_id)
            return removed

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def loaded_ids(self) -> list[str]:
        return list(self._loaded.keys())

    def executor_for(self, model_id: str) -> ThreadPoolExecutor | None:
        """The dedicated thread a loaded model must generate on (see ``_executors``); None when
        the model isn't resident. Callers submit generation here rather than to a shared pool —
        running it on any other thread breaks mlx-vlm's per-thread GPU stream."""
        return self._executors.get(model_id)

    def headroom_bytes(self) -> int | None:
        """Bytes still admittable WITHOUT evicting anything, or None for "unbounded". Lets a
        scheduler (fusion prefetch) ask "would another model fit alongside what's resident?" —
        prefetching must never evict, because the victim may be mid-generation: its memory stays
        held by the running worker while the pool's books say it's free. A full count cap means
        the next load *will* evict, so headroom is 0 regardless of free bytes."""
        cap = self._effective_max()
        if cap is not None and len(self._loaded) >= cap:
            return 0
        if self._ceiling is None:
            return None
        return max(0, self._ceiling - self._loaded_bytes())

    def _harvest_working_memory(self) -> None:
        """Pull each resident engine's measured generation peak into the budget. Done at admission
        time — the one moment the number can change a decision — so engines never need a reference
        back to the pool. Engines that don't measure (VLM) just report nothing and keep the guess."""
        for model_id, engine in self._loaded.items():
            self.record_working_memory(model_id, getattr(engine, "working_memory_bytes", 0) or 0)

    def record_working_memory(self, model_id: str, peak_bytes: int) -> None:
        """Book the working memory a model actually needed to generate (measured by the engine over
        one turn). Keeps the high-water mark: admission has to survive this model's *worst* turn, so
        a later short turn must not make the long one impossible. That means a freak turn inflates
        the budget for the session — deliberate, since the failure it prevents is an OOM kill."""
        if peak_bytes <= 0 or peak_bytes <= self._working_memory.get(model_id, 0):
            return
        self._working_memory[model_id] = peak_bytes
        log.info(
            "working memory measured: %s needs %.1fGB to generate (was budgeting %.1fGB)",
            model_id, peak_bytes / 1e9,
            self._footprint.get(model_id, 0) * (_WORKING_MEMORY_FACTOR - 1) / 1e9,
        )

    def set_mem_ceiling_bytes(self, ceiling: int | None) -> None:
        """Live-update the admission ceiling (GUI Settings edit; the next acquire enforces it, no
        restart). None / <=0 disables byte-admission. Already-loaded models aren't evicted to fit a
        newly-lowered ceiling — it bites on the next load, which is when OOM risk actually arrives."""
        self._ceiling = ceiling if (ceiling and ceiling > 0) else None

    def pin(self, model_id: str) -> None:
        self._pinned.add(model_id)

    def unpin(self, model_id: str) -> None:
        self._pinned.discard(model_id)
