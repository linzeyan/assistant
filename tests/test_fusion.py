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
