"""Fusion: local multi-model panel + judge, exposed as one virtual model.

The user picks the "fusion" model and a turn runs as: each panel model answers the same prompt
*independently and sequentially* (one resident at a time, so 16 GB stays safe — the engine pool
evicts the previous model on the next load), then a judge model reads all candidates and streams
a single synthesized answer. The goal is accuracy, not cost: several models cross-checking each
other beats one model alone on reasoning/research questions. First cut is text-only — the panel
gets no tools (a tool-using fusion is a much larger design, left for later).

Config (enabled / panel / judge) is persisted so it survives restarts and can be changed live
from the API; the model service treats fusion as a normal model id everywhere else.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path

log = logging.getLogger("assistant")

FUSION_MODEL_ID = "fusion"

_JUDGE_SYSTEM = (
    "You are the judge of a multi-model panel. Several models answered the same question "
    "independently. Compare them, reconcile disagreements, and produce the single most accurate "
    "answer. Prefer claims best supported by reasoning over majority vote; if a candidate is "
    "wrong on an important point, briefly correct it. Do not mention that you are a judge or "
    "refer to 'candidates' in your final answer — just give the best answer, directly, without "
    "showing any reasoning or thought process."
)

# Every fusion sub-generation (panel and judge) asks the chat template to disable thinking.
# Qwen3.x templates honour it (`enable_thinking=False` renders an empty <think/> block instead
# of an OPEN `<think>` — with the open form the model's whole output is untagged reasoning that
# nothing downstream can collapse, and a tight max_tokens truncates before the answer even
# starts). Templates without the variable (Mixtral, gemma) simply ignore it.
_NO_THINKING = {"enable_thinking": False}


def _strip_thinking(text: str) -> str:
    """Remove model reasoning from a panel candidate so the judge reads answers, not noise.

    Covers the three shapes seen in practice: tagged ``<think>…</think>`` blocks; the Qwen3.x
    template-opened form where the output has NO opening tag and everything before a bare
    ``</think>`` is reasoning; a dangling ``<think>``/``<|channel>`` when generation was
    truncated mid-thought; and gemma's ``<|channel>…<channel|>`` reasoning channels."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
    if "<|channel>" in text:
        text = text.split("<|channel>", 1)[0]
    return text.strip()


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            return m["content"]
    return ""


def _judge_messages(messages: list[dict], candidates: list[tuple[str, str]]) -> list[dict]:
    question = _last_user(messages)
    blocks = "\n\n".join(
        f"### Candidate {i} (model: {model})\n{text.strip() or '(no answer)'}"
        for i, (model, text) in enumerate(candidates, 1)
    )
    user = (
        f"## Question\n{question}\n\n"
        f"## Panel answers\n{blocks}\n\n"
        "## Task\nSynthesize the single most accurate answer to the question."
    )
    return [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}]


class FusionEngine:
    def __init__(
        self,
        path: Path,
        *,
        enabled: bool = False,
        panel: list[str] | None = None,
        judge: str | None = None,
    ):
        self._path = Path(path)
        self._enabled = enabled
        self._panel = list(panel or [])
        self._judge = judge
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text())
                self._enabled = bool(raw.get("enabled", self._enabled))
                self._panel = list(raw.get("panel", self._panel))
                self._judge = raw.get("judge", self._judge)
            except (OSError, ValueError):
                log.warning("could not read fusion config at %s", self._path)

    @property
    def enabled(self) -> bool:
        # Only actually usable with at least one panel model AND a judge — otherwise the virtual
        # model would be offered but fail at generation time.
        return bool(self._enabled and self._panel and self._judge)

    @property
    def config(self) -> dict:
        return {"enabled": self._enabled, "panel": list(self._panel), "judge": self._judge}

    def configure(
        self,
        *,
        enabled: bool | None = None,
        panel: list[str] | None = None,
        judge: str | None = None,
    ) -> dict:
        if enabled is not None:
            self._enabled = bool(enabled)
        if panel is not None:
            self._panel = list(panel)
        if judge is not None:
            self._judge = judge or None
        try:
            self._path.write_text(json.dumps(self.config))
        except OSError:
            log.exception("could not persist fusion config to %s", self._path)
        return self.config

    @staticmethod
    def _scheduling(service) -> bool:
        """Whether ``service`` exposes the size-aware planning API (the native MLX service does;
        omlx / test fakes may not). Without it the panel just runs sequentially as before."""
        return all(
            hasattr(service, a)
            for a in ("loaded_model_ids", "estimate_bytes", "headroom_bytes", "load", "unload")
        )

    @staticmethod
    async def _plan_order(
        service, panel: list[str], judge: str | None
    ) -> tuple[list[str], dict[str, int]]:
        """Load/unload order for the panel: already-resident models first (their load is a paid
        cost — run them before anything evicts them), then the rest largest-first, so the big
        loads happen while memory is emptiest and each unload-after-use frees room for the
        progressively smaller prefetches (and finally the judge). Sizes include the judge — its
        prefetch must pass the same headroom check as everyone else's."""
        resident = [m for m in service.loaded_model_ids() if m in panel]
        sizes = {m: await service.estimate_bytes(m) for m in {*panel, judge} if m}
        rest = sorted(
            (m for m in panel if m not in resident), key=lambda m: -sizes.get(m, 0)
        )
        return resident + rest, sizes

    async def answer(
        self, service, messages: list[dict], *, max_tokens: int = 1024, **_ignored
    ) -> AsyncIterator[dict]:
        """Run panel → judge, yielding loop-compatible events: tool_progress for the panel/judge
        phases and the judge's text deltas as the answer. ``service`` is the model service (used
        for each sub-generation); the panel runs with no tools.

        Memory-aware scheduling (when the service supports it): panel order is resident-first
        then largest-first; while one model generates, the next is prefetched in the background
        *iff it fits in the current headroom* (prefetch must never evict — the generating model
        is memory the ceiling thinks is free the moment it's evicted, but isn't); each panel
        model is unloaded right after its candidate is collected (a panel member runs once per
        turn) unless the user had it loaded before the turn or it doubles as the judge — which
        always runs last, over the freed memory."""
        panel, judge = list(self._panel), self._judge
        n = len(panel)
        sched = self._scheduling(service)
        if sched:
            order, sizes = await self._plan_order(service, panel, judge)
            pre_loaded = set(service.loaded_model_ids())
            log.info(
                "fusion schedule: %s -> judge %s",
                " -> ".join(f"{m} (~{sizes.get(m, 0) / 1e9:.1f}GB)" for m in order), judge,
            )
        else:
            order, sizes, pre_loaded = panel, {}, set()

        async def _load_quiet(model_id: str) -> None:
            # Prefetch failures aren't fatal here — the model's own turn retries the load and
            # reports the real error through the normal skip path.
            try:
                await service.load(model_id)
            except Exception as e:  # noqa: BLE001
                log.debug("fusion prefetch of %s failed (will retry at its turn): %s", model_id, e)

        def _start_prefetch(model_id: str | None) -> asyncio.Task | None:
            if not (sched and model_id):
                return None
            headroom = service.headroom_bytes()
            need = sizes.get(model_id, 0)
            if headroom is not None and need > headroom:
                return None  # doesn't fit alongside what's resident — load inline at its turn
            log.info("fusion prefetch: %s (~%.1fGB, headroom %s)", model_id, need / 1e9,
                     "∞" if headroom is None else f"{headroom / 1e9:.1f}GB")
            return asyncio.create_task(_load_quiet(model_id))

        candidates: list[tuple[str, str]] = []
        failures: list[tuple[str, str]] = []
        # Strong refs to every in-flight prefetch: asyncio keeps only weak refs to tasks, and a
        # collected task would abort its load midway (and strand the pool lock's queue).
        prefetches: set[asyncio.Task] = set()
        for i, model in enumerate(order, 1):
            yield {
                "type": "tool_progress",
                "name": "fusion",
                "fraction": (i - 1) / (n + 1),
                "label": f"panel {i}/{n}: {model}",
            }
            # A panel model that won't load (e.g. an arch too new for the installed mlx-lm) must
            # not sink the whole turn — skip it, keep its slot's progress, and let the judge work
            # with the survivors. Only a fully empty panel is fatal.
            try:
                if sched:
                    # Ensure THIS model is resident before starting the next prefetch, so the
                    # prefetch can't win the pool lock and delay the answer we're waiting on.
                    await service.load(model)
                task = _start_prefetch(
                    order[i] if i < n else (judge if judge not in pre_loaded else None)
                )
                if task is not None:
                    prefetches.add(task)
                    task.add_done_callback(prefetches.discard)
                parts: list[str] = []
                async for ev in service.stream_chat(
                    messages, model, max_tokens=max_tokens, chat_template_kwargs=_NO_THINKING
                ):
                    if ev.get("type") == "text":
                        parts.append(ev["content"])
                # Sanitize even with thinking disabled: some templates ignore the kwarg and a
                # truncated thought would otherwise reach the judge as the whole "answer".
                candidates.append((model, _strip_thinking("".join(parts))))
            except Exception as e:  # noqa: BLE001 — any per-model failure is isolated here
                log.warning("fusion panel model %s failed, skipping: %s", model, e)
                failures.append((model, str(e)))
                yield {
                    "type": "tool_progress",
                    "name": "fusion",
                    "fraction": i / (n + 1),
                    "label": f"skipped {model} (failed to load)",
                }
            # Used panel models are done for this turn — release their memory for the next
            # prefetch / the judge. Keep anything the user had loaded before the turn (their
            # chat model) and the judge itself (about to be used).
            if sched and model != judge and model not in pre_loaded and model not in order[i:]:
                try:
                    await service.unload(model)
                except Exception:  # noqa: BLE001 — freeing memory is best-effort
                    log.debug("fusion could not unload %s after use", model)
        if not candidates:
            detail = "; ".join(f"{m}: {e}" for m, e in failures) or "no panel models configured"
            raise RuntimeError(f"Fusion: every panel model failed — {detail}")
        yield {
            "type": "tool_progress",
            "name": "fusion",
            "fraction": n / (n + 1),
            "label": f"judge: {judge}",
        }
        for task in list(prefetches):
            await task  # drain in-flight prefetches (normally just the judge's) before streaming
        async for ev in service.stream_chat(
            _judge_messages(messages, candidates), judge, max_tokens=max_tokens,
            chat_template_kwargs=_NO_THINKING,
        ):
            yield ev  # judge text deltas flow straight through as the answer
