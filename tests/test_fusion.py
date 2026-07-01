"""Fusion panel+judge engine and its model-service integration (Spring 3 S3.3)."""

from __future__ import annotations

from assistant.agent.fusion import FusionEngine
from assistant.models.mlx_service import MlxModelService


class _FakeService:
    """A model service that echoes a deterministic per-model answer, recording each call."""

    def __init__(self):
        self.calls: list[tuple[str, list[dict]]] = []

    async def stream_chat(self, messages, model, tools=None, **params):
        self.calls.append((model, messages))
        yield {"type": "text", "content": f"answer-from-{model}"}


async def test_fusion_answer_runs_panel_then_judge(tmp_path):
    eng = FusionEngine(tmp_path / "f.json", enabled=True, panel=["a", "b"], judge="j")
    svc = _FakeService()
    events = [
        ev async for ev in eng.answer(svc, [{"role": "user", "content": "Q"}], max_tokens=64)
    ]
    # Panel models run sequentially, then the judge.
    assert [m for m, _ in svc.calls] == ["a", "b", "j"]
    # A progress tick per panel model + one for the judge phase.
    assert sum(1 for e in events if e["type"] == "tool_progress") == 3
    # The answer streamed back is the judge's text.
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "answer-from-j"
    # The judge was given both candidates to synthesize from.
    judge_prompt = svc.calls[-1][1][-1]["content"]
    assert "answer-from-a" in judge_prompt and "answer-from-b" in judge_prompt


class _FlakyService:
    """Echoes an answer per model, but raises for any model id in ``broken`` — mimics a panel
    model whose architecture the installed mlx-lm can't load."""

    def __init__(self, broken: set[str]):
        self.broken = broken
        self.calls: list[str] = []

    async def stream_chat(self, messages, model, tools=None, **params):
        self.calls.append(model)
        if model in self.broken:
            raise RuntimeError(f"Model type {model} not supported")
        yield {"type": "text", "content": f"answer-from-{model}"}


async def test_fusion_skips_a_failing_panel_model(tmp_path):
    # A single unloadable panel model must not sink the whole turn — this is the real 500 we hit:
    # one bad model raising aborted fusion entirely. The judge still runs over the survivors.
    eng = FusionEngine(tmp_path / "f.json", enabled=True, panel=["a", "bad", "c"], judge="j")
    svc = _FlakyService(broken={"bad"})
    events = [ev async for ev in eng.answer(svc, [{"role": "user", "content": "Q"}])]

    # Every model was attempted, then the judge still ran despite "bad" raising.
    assert svc.calls == ["a", "bad", "c", "j"]
    text = "".join(e["content"] for e in events if e["type"] == "text")
    assert text == "answer-from-j"
    # The skip surfaces as a progress tick so the user isn't left wondering.
    assert any(
        "skipped bad" in e.get("label", "") for e in events if e["type"] == "tool_progress"
    )


async def test_fusion_all_panel_failed_is_a_loud_error(tmp_path):
    import pytest

    eng = FusionEngine(tmp_path / "f.json", enabled=True, panel=["x", "y"], judge="j")
    svc = _FlakyService(broken={"x", "y"})
    with pytest.raises(RuntimeError, match="every panel model failed"):
        [ev async for ev in eng.answer(svc, [{"role": "user", "content": "Q"}])]


def test_fusion_enabled_requires_panel_and_judge(tmp_path):
    p = tmp_path / "f.json"
    assert not FusionEngine(p, enabled=True, panel=[], judge=None).enabled
    assert not FusionEngine(p, enabled=True, panel=["a"], judge=None).enabled
    assert not FusionEngine(p, enabled=False, panel=["a"], judge="j").enabled
    assert FusionEngine(p, enabled=True, panel=["a"], judge="j").enabled


def test_fusion_configure_persists(tmp_path):
    p = tmp_path / "f.json"
    FusionEngine(p).configure(enabled=True, panel=["a", "b"], judge="j")
    reloaded = FusionEngine(p)
    assert reloaded.enabled
    assert reloaded.config == {"enabled": True, "panel": ["a", "b"], "judge": "j"}


async def test_service_lists_fusion_when_enabled(tmp_path):
    eng = FusionEngine(tmp_path / "f.json", enabled=True, panel=["a"], judge="j")
    svc = MlxModelService(models_dir=tmp_path, available_override=True, fusion=eng)
    ids = [m.id for m in await svc.list_models()]
    assert "fusion" in ids
    # Disabled fusion is not offered.
    eng.configure(enabled=False)
    assert "fusion" not in [m.id for m in await svc.list_models()]


async def test_service_load_fusion_is_noop(tmp_path):
    eng = FusionEngine(tmp_path / "f.json", enabled=True, panel=["a"], judge="j")
    svc = MlxModelService(models_dir=tmp_path, available_override=True, fusion=eng)
    await svc.load("fusion")  # virtual model: must not raise "unknown model"
