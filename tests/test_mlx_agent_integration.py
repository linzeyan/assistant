"""End-to-end: the agent loop driving tools through the native MLX backend.

Proves parity with the omlx path — a model that emits tool-call *text* (the way
mlx-lm does) drives a real think -> tool -> observe -> answer cycle, with no special
handling in the loop.
"""

from __future__ import annotations

from assistant.agent.llm_client import AsyncLLM
from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.models.mlx_engine import MlxEnginePool
from assistant.models.mlx_service import MlxModelService
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import ToolContext


class ScriptedEngine:
    """A loaded engine whose Nth generation returns the Nth scripted token list."""

    def __init__(self, scripts):
        self._scripts = scripts
        self._i = 0

    def stream_text(self, messages, **kwargs):
        script = self._scripts[self._i]
        self._i += 1
        yield from script


def _make_model(tmp_path, name):
    import json

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (d / "model.safetensors").write_bytes(b"\x00")  # discovery requires real weights


async def test_full_tool_cycle_through_native_backend(tmp_path):
    (tmp_path / "x.txt").write_text("FILE BODY")
    _make_model(tmp_path, "qwen")

    engine = ScriptedEngine(
        [
            # Turn 1: the model "decides" to call read_file (emitted as text).
            ['<tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>'],
            # Turn 2: having seen the tool result, it answers.
            ["The file says FILE BODY."],
        ]
    )
    pool = MlxEnginePool(max_loaded=1, loader=lambda _path: engine)
    svc = MlxModelService(
        models_dir=tmp_path,
        include_hf_cache=False,
        pool=pool,
        available_override=True,
    )
    await svc.start()

    loop = AgentLoop(
        AsyncLLM(svc),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
    )
    events = [
        e async for e in loop.run(Session(id="s"), "read x.txt", "qwen")
    ]

    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_result" in types and types[-1] == "done"

    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["ok"] and tool_result["content"] == "FILE BODY"

    answer = "".join(e["content"] for e in events if e["type"] == "assistant_delta")
    assert answer == "The file says FILE BODY."
