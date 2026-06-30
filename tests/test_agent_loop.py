import asyncio

from assistant.agent.loop import AgentLoop
from assistant.agent.session import Session
from assistant.tools import build_registry
from assistant.tools.approval import PolicyApprover
from assistant.tools.base import Tool, ToolContext, ToolResult
from assistant.tools.registry import ToolRegistry


class FakeLLM:
    """Scripted LLM: returns a preset event list per call (one call == one turn)."""

    def __init__(self, turns: list[list[dict]]):
        self._turns = turns
        self._i = 0

    def stream_chat(self, messages, model, tools=None, **params):
        events = self._turns[self._i]
        self._i += 1

        async def gen():
            for e in events:
                yield e

        return gen()


async def _collect(agen):
    return [e async for e in agen]


def _loop(llm, tmp_path, approval_required: bool) -> AgentLoop:
    return AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=approval_required),
        ToolContext(cwd=tmp_path),
    )


async def test_tool_progress_events_streamed(tmp_path):
    # WHY: a long tool (video denoising) must not look frozen. Ticks reported via
    # ctx.on_progress are surfaced as tool_progress events, ordered strictly between the
    # tool_call and its tool_result so a gateway can render a live progress bar.
    async def slow(args, ctx):
        ctx.on_progress(0.5, "1/2")
        ctx.on_progress(1.0, "2/2")
        return ToolResult(True, "ok")

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="slow",
            description="reports progress",
            parameters={"type": "object", "properties": {}},
            handler=slow,
        )
    )
    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "slow", "arguments": {}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    loop = AgentLoop(
        llm, reg, PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path)
    )
    events = await _collect(loop.run(Session(id="s1"), "go", "m"))

    progress = [e for e in events if e["type"] == "tool_progress"]
    assert [round(e["fraction"], 2) for e in progress] == [0.5, 1.0]
    assert progress[0] == {"type": "tool_progress", "id": "c1", "name": "slow",
                           "fraction": 0.5, "label": "1/2"}
    types = [e["type"] for e in events]
    assert types.index("tool_call") < types.index("tool_progress") < types.index("tool_result")
    # on_progress is scoped to the run and cleared afterwards (no leak into later tools).
    assert loop._ctx.on_progress is None


async def test_heartbeat_emitted_for_silent_long_tool(tmp_path):
    # WHY: a long tool that reports no progress of its own (bash, web fetch) must not look
    # frozen — the loop emits an elapsed-time heartbeat (fraction < 0) so a gateway can show
    # "working…".
    async def slow(args, ctx):
        await asyncio.sleep(0.3)  # silent: never calls ctx.on_progress
        return ToolResult(True, "ok")

    reg = ToolRegistry()
    reg.register(Tool("slow", "", {"type": "object", "properties": {}}, slow))
    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "slow", "arguments": {}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    loop = AgentLoop(llm, reg, PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path))
    loop._HEARTBEAT_SECS = 0.05  # fire quickly instead of the 5s default
    events = await _collect(loop.run(Session(id="hb"), "go", "m"))
    hb = [e for e in events if e["type"] == "tool_progress" and e["fraction"] < 0]
    assert hb and ":" in hb[0]["label"]  # elapsed "m:ss"


async def test_turn_diff_emitted_for_file_edits(tmp_path):
    # WHY: code results are first-class — a turn that writes/edits files emits a turn_diff
    # event (after the edits, before done) so a gateway can return the diff to the user.
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "hello.py", "content": "print('hi')\n"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "done"}],
        ]
    )
    events = await _collect(
        _loop(llm, tmp_path, approval_required=False).run(Session(id="s1"), "make it", "m")
    )
    td = next(e for e in events if e["type"] == "turn_diff")
    assert td["files"][0]["path"] == "hello.py"
    assert td["files"][0]["status"] == "added"
    assert "print('hi')" in td["diff"]
    types = [e["type"] for e in events]  # ordered: result → diff → done
    assert types.index("tool_result") < types.index("turn_diff") < types.index("done")


async def test_run_cwd_override_directs_tools_and_diff(tmp_path):
    # WHY: workspace is per-conversation — run(cwd=...) overrides where the edit tools operate
    # (and where the turn diff is computed) for this turn only, without mutating the shared
    # ToolContext other turns/entries share.
    other = tmp_path / "proj"
    other.mkdir()
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "a.py", "content": "x = 1\n"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "done"}],
        ]
    )
    loop = _loop(llm, tmp_path, approval_required=False)  # base cwd = tmp_path
    events = await _collect(loop.run(Session(id="s1"), "go", "m", cwd=str(other)))
    assert (other / "a.py").read_text() == "x = 1\n"  # wrote into the override dir
    assert not (tmp_path / "a.py").exists()  # not the base cwd
    td = next(e for e in events if e["type"] == "turn_diff")
    assert td["files"][0]["path"] == "a.py"  # path shown relative to the override cwd
    assert loop._ctx.cwd == tmp_path  # shared ctx left untouched


async def test_turn_diff_includes_shell_created_files(tmp_path):
    # WHY: write/edit snapshots can't see files the bash tool creates. In a git repo the loop
    # discovers them and folds them into the turn diff, so "make a file via shell" still
    # returns what changed.
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "seed.txt").write_text("x\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "init"], check=True)

    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [
                {"id": "c1", "name": "bash", "arguments": {"command": "echo hi > made.txt"}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    events = await _collect(
        _loop(llm, tmp_path, approval_required=False).run(Session(id="sh"), "make a file", "m")
    )
    td = next(e for e in events if e["type"] == "turn_diff")
    # macOS tmp lives under a /private symlink so the path may be absolute; match by suffix.
    assert any(f["path"].endswith("made.txt") and f["status"] == "added" for f in td["files"])


async def test_tool_then_final_answer(tmp_path):
    (tmp_path / "x.txt").write_text("FILE BODY")
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {"id": "c1", "name": "read_file", "arguments": {"path": "x.txt"}}
                    ],
                }
            ],
            [{"type": "text", "content": "the file says FILE BODY"}],
        ]
    )
    session = Session(id="s1")
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        session, "read x.txt", "m"
    ))

    assert [e["type"] for e in events] == [
        "tool_call",
        "tool_result",
        "assistant_delta",
        "done",
    ]
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and tr["content"] == "FILE BODY"
    # Tool output and final answer are persisted so the next turn has context.
    assert any(m.get("role") == "tool" and m["content"] == "FILE BODY" for m in session.messages)
    assert session.messages[-1] == {
        "role": "assistant",
        "content": "the file says FILE BODY",
    }


async def test_approval_gate_blocks_mutation(tmp_path):
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "data"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "could not write"}],
        ]
    )
    # approval_required=True + non-interactive approver => denied.
    events = await _collect(_loop(llm, tmp_path, approval_required=True).run(
        Session(id="s2"), "write a file", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False and "requires approval" in tr["content"]
    assert not (tmp_path / "out.txt").exists()  # mutation actually blocked


async def test_approval_disabled_allows_mutation(tmp_path):
    llm = FakeLLM(
        [
            [
                {
                    "type": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "write_file",
                            "arguments": {"path": "out.txt", "content": "data"},
                        }
                    ],
                }
            ],
            [{"type": "text", "content": "wrote it"}],
        ]
    )
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        Session(id="s3"), "write a file", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] and (tmp_path / "out.txt").read_text() == "data"


async def test_unknown_tool_reported(tmp_path):
    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [{"id": "c1", "name": "nope", "arguments": {}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(
        Session(id="s4"), "x", "m"
    ))
    tr = next(e for e in events if e["type"] == "tool_result")
    assert tr["ok"] is False and "unknown tool" in tr["content"]


async def test_forwards_max_output_tokens_to_llm(tmp_path):
    """The configured per-turn ceiling must actually reach the engine — the engine's own
    default was 1024, which silently truncated long answers (bug #1)."""
    captured: dict = {}

    class CapturingLLM:
        def stream_chat(self, messages, model, tools=None, **params):
            captured.update(params)

            async def gen():
                yield {"type": "text", "content": "hi"}

            return gen()

    loop = AgentLoop(
        CapturingLLM(),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
        max_output_tokens=2048,
    )
    await _collect(loop.run(Session(id="mt"), "hello", "m"))
    assert captured.get("max_tokens") == 2048


async def test_injects_working_directory_into_user_turn(tmp_path):
    """A local model has no idea where it is, so a bare 'git diff' gets a guessed path. The
    per-turn working directory (here a run(cwd=...) override, as /cd sets) must reach the model
    on the latest user turn so shell/file commands execute in the workspace."""
    captured: dict = {}

    class CapturingLLM:
        def stream_chat(self, messages, model, tools=None, **params):
            captured["messages"] = messages

            async def gen():
                yield {"type": "text", "content": "ok"}

            return gen()

    other = tmp_path / "proj"
    other.mkdir()
    loop = AgentLoop(
        CapturingLLM(),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
    )
    await _collect(loop.run(Session(id="w1"), "git diff", "m", cwd=str(other)))
    last_user = [m for m in captured["messages"] if m["role"] == "user"][-1]
    assert "Working directory:" in last_user["content"]
    assert str(other) in last_user["content"]  # the /cd override, not the base cwd


async def test_injects_current_date_into_user_turn(tmp_path):
    """A local model has no clock; the current date rides the latest user message (not the
    system prompt) so 'today' isn't hallucinated from the training cutoff. Verified by
    capturing exactly what reaches the engine."""
    captured: dict = {}

    class CapturingLLM:
        def stream_chat(self, messages, model, tools=None, **params):
            captured["messages"] = messages

            async def gen():
                yield {"type": "text", "content": "ok"}

            return gen()

    loop = AgentLoop(
        CapturingLLM(),
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
    )
    session = Session(id="d1")
    await _collect(loop.run(session, "今天美股狀況", "m"))

    last_user = [m for m in captured["messages"] if m["role"] == "user"][-1]
    assert "current-datetime" in last_user["content"]
    assert "今天美股狀況" in last_user["content"]  # original text preserved
    # Stored history stays clean — injection is send-time only.
    assert any(
        m.get("role") == "user" and m["content"] == "今天美股狀況" for m in session.messages
    )


async def test_view_image_skipped_for_freshly_generated_image(tmp_path):
    # The agent sometimes calls view_image on an image it just generated; that only loads a
    # vision model to redescribe output the user already has. The loop must short-circuit it.
    img = tmp_path / "gen.png"

    class _Images:
        def available(self):
            return True

        async def generate_image(self, prompt, **kw):
            img.write_bytes(b"\x89PNG")
            return img

    class _Vision:
        def __init__(self):
            self.calls = 0

        def available(self):
            return True

        async def describe(self, paths, q):
            self.calls += 1
            return "described"

    vision = _Vision()
    llm = FakeLLM(
        [
            [{"type": "tool_calls", "tool_calls": [
                {"id": "c1", "name": "generate_image", "arguments": {"prompt": "a nebula"}}]}],
            [{"type": "tool_calls", "tool_calls": [
                {"id": "c2", "name": "view_image", "arguments": {"path": str(img)}}]}],
            [{"type": "text", "content": "done"}],
        ]
    )
    loop = AgentLoop(
        llm,
        build_registry(),
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path, images=_Images(), vision=vision),
    )
    events = await _collect(loop.run(Session(id="vi"), "make a nebula and look", "m"))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[0]["name"] == "generate_image" and results[0]["ok"]
    view_res = results[1]
    assert view_res["name"] == "view_image" and view_res["ok"]
    assert "already generated" in view_res["content"]
    assert vision.calls == 0  # the vision model was never loaded


async def test_max_iters_honored_and_configurable(tmp_path):
    # WHY: Spring4 SB.3 measured a skill-driven turn (skill_view + reproduce + read + git log +
    # git show + fix + regression test) running well past the old default of 8 — the loop cut it
    # off mid-investigation. The per-turn budget must be raisable AND honored exactly, and when it
    # is reached the loop must stop LOUD (a ceiling error), never silently truncate the work.
    async def noop(args, ctx):
        return ToolResult(True, "ok")

    reg = ToolRegistry()
    reg.register(
        Tool(
            name="noop",
            description="does nothing",
            parameters={"type": "object", "properties": {}},
            handler=noop,
        )
    )
    # Every LLM turn emits another tool call, so the loop only ever stops at the ceiling.
    tool_turn = [{"type": "tool_calls", "tool_calls": [{"id": "c", "name": "noop", "arguments": {}}]}]
    llm = FakeLLM([tool_turn] * 50)
    loop = AgentLoop(
        llm,
        reg,
        PolicyApprover(approval_required=False),
        ToolContext(cwd=tmp_path),
        max_iters=12,  # a raised budget, as the app now wires from settings.max_tool_iters
    )
    events = await _collect(loop.run(Session(id="s"), "go", "m"))
    assert sum(e["type"] == "tool_call" for e in events) == 12  # ran the full raised budget
    err = next(e for e in events if e["type"] == "error")
    assert "12" in err["detail"]  # and surfaced the ceiling, not a silent stop


async def test_update_plan_emits_plan_event_with_normalized_steps(tmp_path):
    # WHY (SA.3): the agent's checklist must reach the UI as a structured `plan` event — but the
    # full list must NOT be persisted into history (only the tool's short ack), so a stale
    # checklist isn't re-fed every iteration (token bloat on small models).
    llm = FakeLLM([
        [{"type": "tool_calls", "tool_calls": [{
            "id": "p1", "name": "update_plan",
            "arguments": {"steps": [
                {"title": "read the file", "status": "completed"},
                {"title": "  ", "status": "pending"},  # empty title is dropped
                {"title": "fix the bug", "status": "weird"},  # bad status → pending
            ]},
        }]}],
        [{"type": "text", "content": "done"}],
    ])
    session = Session(id="plan1")
    events = await _collect(_loop(llm, tmp_path, approval_required=False).run(session, "go", "m"))
    plan = next(e for e in events if e["type"] == "plan")
    assert plan["steps"] == [
        {"title": "read the file", "status": "completed"},
        {"title": "fix the bug", "status": "pending"},
    ]
    # History carries only the short ack, never the full checklist (no step titles leak in).
    tool_msgs = [m for m in session.messages if m.get("role") == "tool"]
    assert tool_msgs and "plan updated" in tool_msgs[-1]["content"]
    assert "read the file" not in tool_msgs[-1]["content"]
