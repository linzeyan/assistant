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
    # Every LLM turn emits another tool call, so the loop only ever stops at the ceiling. The args
    # differ each turn (distinct {"i": n}) so the B2 thrash guard — which aborts on *identical*
    # repeats — doesn't pre-empt the max_iters ceiling this test is exercising.
    llm = FakeLLM([
        [{"type": "tool_calls",
          "tool_calls": [{"id": f"c{i}", "name": "noop", "arguments": {"i": i}}]}]
        for i in range(50)
    ])
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


async def test_per_request_max_iters_overrides_default(tmp_path):
    # H7: a single turn can override the loop's configured ceiling without changing the global
    # default. A non-positive / None override falls back to the default (can't disable the backstop).
    async def noop(args, ctx):
        return ToolResult(True, "ok")

    reg = _reg_with(_tool("noop", noop))

    def _llm():  # fresh scripted LLM each run (FakeLLM is single-use, one entry per turn)
        return FakeLLM([
            [{"type": "tool_calls",
              "tool_calls": [{"id": f"c{i}", "name": "noop", "arguments": {"i": i}}]}]
            for i in range(50)
        ])

    def _loop_with(llm):
        return AgentLoop(llm, reg, PolicyApprover(approval_required=False),
                         ToolContext(cwd=tmp_path), max_iters=12)

    # Override below the default → the turn stops at 3.
    events = await _collect(_loop_with(_llm()).run(Session(id="s"), "go", "m", max_iters=3))
    assert sum(e["type"] == "tool_call" for e in events) == 3
    assert "3" in next(e for e in events if e["type"] == "error")["detail"]

    # None and 0 both fall back to the configured default (12).
    for override in (None, 0):
        evs = await _collect(_loop_with(_llm()).run(Session(id="s"), "go", "m", max_iters=override))
        assert sum(e["type"] == "tool_call" for e in evs) == 12


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


async def test_turn_timeout_aborts_between_iterations(tmp_path):
    # WHY (B1): a runaway tool-call loop must self-abort on the wall-clock budget — between
    # iterations, with a loud error — rather than running to the iteration ceiling or unbounded.
    reg = ToolRegistry()

    async def noop(args, ctx):
        return ToolResult(True, "ok")

    reg.register(Tool(
        name="noop", description="", parameters={"type": "object", "properties": {}}, handler=noop
    ))

    class SlowToolLLM:
        # Each turn takes real time, then asks for another tool call, so the loop keeps iterating.
        def stream_chat(self, messages, model, tools=None, **params):
            async def gen():
                await asyncio.sleep(0.05)
                yield {"type": "tool_calls",
                       "tool_calls": [{"id": "c", "name": "noop", "arguments": {}}]}
            return gen()

    loop = AgentLoop(
        SlowToolLLM(), reg, PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path),
        max_iters=50, turn_timeout_s=0.01,
    )
    events = await _collect(loop.run(Session(id="t"), "go", "m"))
    err = next(e for e in events if e["type"] == "error")
    assert "time limit" in err["detail"]  # timed out, not the iteration ceiling
    assert "max tool iterations" not in err["detail"]
    assert sum(e["type"] == "tool_call" for e in events) < 50  # aborted well before the ceiling


async def test_no_turn_timeout_runs_to_completion(tmp_path):
    # Default (None) must not impose any limit — a normal turn answers as before.
    llm = FakeLLM([[{"type": "text", "content": "done"}]])
    loop = AgentLoop(
        llm, build_registry(), PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path),
        turn_timeout_s=None,
    )
    events = await _collect(loop.run(Session(id="t2"), "hi", "m"))
    assert any(e["type"] == "done" for e in events)
    assert not any(e["type"] == "error" for e in events)


# --- B2: thrash / no-progress guardrails -------------------------------------------------

from assistant.agent.loop import _THRASH_REPEAT_CAP  # noqa: E402


def _reg_with(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


def _tool(name, handler, *, needs_approval=False):
    return Tool(name=name, description="", parameters={"type": "object", "properties": {}},
                handler=handler, needs_approval=needs_approval)


def _call(cid, name, args):
    return [{"type": "tool_calls", "tool_calls": [{"id": cid, "name": name, "arguments": args}]}]


async def test_b2_idempotent_repeat_is_replayed_not_rerun(tmp_path):
    # An identical read-only call makes no progress: the handler runs once and the repeat replays
    # the cached result (annotated), instead of spending another tool round-trip on the same read.
    ran = []

    async def reader(args, ctx):
        ran.append(args["path"])
        return ToolResult(True, "FILE CONTENTS")

    llm = FakeLLM([
        _call("c1", "reader", {"path": "a"}),
        _call("c2", "reader", {"path": "a"}),  # identical → replayed
        [{"type": "text", "content": "done"}],
    ])
    loop = AgentLoop(llm, _reg_with(_tool("reader", reader)),
                     PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path))
    events = await _collect(loop.run(Session(id="s"), "go", "m"))
    assert ran == ["a"]  # handler invoked exactly once
    results = [e for e in events if e["type"] == "tool_result" and e["name"] == "reader"]
    assert len(results) == 2  # both calls still surfaced a result
    assert "repeated call" in results[1]["content"] and "FILE CONTENTS" in results[1]["content"]
    assert not any(e["type"] == "error" for e in events)


async def test_b2_aborts_on_repeated_call_thrash(tmp_path):
    # The model keeps emitting the same call despite the "you're repeating" nudge → stuck. Abort
    # loudly at the cap instead of burning the whole iteration budget.
    async def reader(args, ctx):
        return ToolResult(True, "x")

    turns = [_call(f"c{i}", "reader", {"path": "a"}) for i in range(6)]
    loop = AgentLoop(FakeLLM(turns), _reg_with(_tool("reader", reader)),
                     PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path),
                     max_iters=10)
    events = await _collect(loop.run(Session(id="s"), "go", "m"))
    err = next(e for e in events if e["type"] == "error")
    assert "repeated" in err["detail"] and "max tool iterations" not in err["detail"]
    # Stopped at the cap, not after all six iterations.
    assert len([e for e in events if e["type"] == "tool_result"]) <= _THRASH_REPEAT_CAP + 1


async def test_b2_mutation_invalidates_replay_cache(tmp_path):
    # A re-read after a mutation must RUN (state changed), not replay the pre-mutation result.
    reads = []

    async def reader(args, ctx):
        reads.append(1)
        return ToolResult(True, f"v{len(reads)}")

    async def writer(args, ctx):
        return ToolResult(True, "written")

    reg = _reg_with(_tool("reader", reader), _tool("writer", writer, needs_approval=True))
    llm = FakeLLM([
        _call("r1", "reader", {"path": "a"}),
        _call("w1", "writer", {"path": "a"}),  # mutation → invalidates the read cache
        _call("r2", "reader", {"path": "a"}),  # must re-run, not replay
        [{"type": "text", "content": "done"}],
    ])
    loop = AgentLoop(llm, reg, PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path))
    await _collect(loop.run(Session(id="s"), "go", "m"))
    assert len(reads) == 2  # the second read ran for real


async def test_b2_distinct_calls_never_thrash(tmp_path):
    # Healthy turns chain several DIFFERENT calls — these must all run, none deduped, no abort.
    seen = []

    async def reader(args, ctx):
        seen.append(args["path"])
        return ToolResult(True, "ok")

    turns = [_call(f"c{i}", "reader", {"path": f"p{i}"}) for i in range(5)]
    turns.append([{"type": "text", "content": "done"}])
    loop = AgentLoop(FakeLLM(turns), _reg_with(_tool("reader", reader)),
                     PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path),
                     max_iters=10)
    events = await _collect(loop.run(Session(id="s"), "go", "m"))
    assert seen == ["p0", "p1", "p2", "p3", "p4"]
    assert not any(e["type"] == "error" for e in events)


# --- G: modality schema-time gate (omit unavailable tools) -------------------------------

from assistant.tools.base import service_available  # noqa: E402


def test_g_service_available_predicate(tmp_path):
    check = service_available("vision")
    assert check(ToolContext(cwd=tmp_path)) is False  # no service on the context

    class OK:
        def available(self):
            return True

    class NotInstalled:
        def available(self):
            return False

    assert check(ToolContext(cwd=tmp_path, vision=OK())) is True
    assert check(ToolContext(cwd=tmp_path, vision=NotInstalled())) is False


async def test_g_schema_gate_omits_unavailable_tool(tmp_path):
    # A tool whose check_fn reports unavailable is NOT offered to the model — it can't be called and
    # so can't fail at runtime (death mode 3). An ungated tool is always offered.
    async def noop(args, ctx):
        return ToolResult(True, "ok")

    reg = _reg_with(
        _tool("always", noop),
        Tool(name="gated", description="", parameters={"type": "object", "properties": {}},
             handler=noop, check_fn=lambda ctx: False),
    )
    loop = AgentLoop(FakeLLM([]), reg, PolicyApprover(approval_required=False),
                     ToolContext(cwd=tmp_path))
    names = {s["function"]["name"] for s in loop._visible_tool_schemas(ToolContext(cwd=tmp_path))}
    assert "always" in names and "gated" not in names


def test_g_modality_tools_gated_by_context(tmp_path):
    # The real modality tools disappear from the schema when their backend isn't on the context,
    # and reappear when an available service is present. web_search (ungated) is always there.
    loop = AgentLoop(FakeLLM([]), build_registry(), PolicyApprover(approval_required=False),
                     ToolContext(cwd=tmp_path))
    bare = {s["function"]["name"] for s in loop._visible_tool_schemas(ToolContext(cwd=tmp_path))}
    assert {"view_image", "generate_image", "generate_video", "transcribe_audio",
            "text_to_speech"}.isdisjoint(bare)
    assert "web_search" in bare

    class OK:
        def available(self):
            return True

    rich = {
        s["function"]["name"]
        for s in loop._visible_tool_schemas(
            ToolContext(cwd=tmp_path, vision=OK(), images=OK(), video=OK(), audio=OK())
        )
    }
    assert {"view_image", "generate_image", "edit_image", "generate_video",
            "transcribe_audio", "text_to_speech"} <= rich


# --- H3: approval audit log ----------------------------------------------------------------

async def test_h3_approval_audit_logs_allow_and_deny(tmp_path, caplog):
    # H3: every security-relevant (approval-gated) decision leaves one structured audit line with
    # the tool, resource, and allow/deny outcome — independent of which path decided.
    import logging

    async def w(args, ctx):
        return ToolResult(True, "wrote")

    reg = _reg_with(_tool("danger", w, needs_approval=True))

    async def _run(approval_required):
        llm = FakeLLM([_call("c1", "danger", {"path": "x.py"}),
                       [{"type": "text", "content": "done"}]])
        loop = AgentLoop(llm, reg, PolicyApprover(approval_required=approval_required),
                         ToolContext(cwd=tmp_path))
        with caplog.at_level(logging.INFO, logger="assistant"):
            await _collect(loop.run(Session(id="s"), "go", "m"))

    caplog.clear()
    await _run(approval_required=False)  # auto-approved
    assert any("approval audit" in r.getMessage() and "decision=allow" in r.getMessage()
               and "danger" in r.getMessage() and "x.py" in r.getMessage()
               for r in caplog.records)

    caplog.clear()
    await _run(approval_required=True)  # non-interactive policy → denied
    assert any("approval audit" in r.getMessage() and "decision=deny" in r.getMessage()
               for r in caplog.records)


async def test_h3_no_audit_for_read_only_tool(tmp_path, caplog):
    # Read-only tools auto-allow and aren't security-interesting — they must not spam the audit log.
    import logging

    async def ok(args, ctx):
        return ToolResult(True, "ok")

    reg = _reg_with(_tool("noop", ok))  # needs_approval=False
    llm = FakeLLM([_call("c1", "noop", {}), [{"type": "text", "content": "done"}]])
    loop = AgentLoop(llm, reg, PolicyApprover(approval_required=False), ToolContext(cwd=tmp_path))
    with caplog.at_level(logging.INFO, logger="assistant"):
        await _collect(loop.run(Session(id="s"), "go", "m"))
    assert not any("approval audit" in r.getMessage() for r in caplog.records)
