"""DownloadManager lifecycle (N17): progress, cancel, resume-after-restart, retry. The size
lookup and runner are injected so none of this touches the network or spawns a subprocess."""

from __future__ import annotations

import asyncio
import json

from assistant.downloads import DownloadManager


def _manager(tmp_path, *, size_fn=None, runner=None) -> DownloadManager:
    return DownloadManager(
        target_dir=tmp_path / "models",
        state_path=tmp_path / "downloads.json",
        size_fn=size_fn or (lambda _r: 1000),
        runner=runner,
    )


async def test_start_completes_and_reports_full_progress(tmp_path):
    async def runner(repo_id, target, state, cancel):
        state.downloaded_bytes = 1000

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    await mgr._tasks["org/model"]
    [item] = mgr.snapshot()
    assert item["status"] == "done"
    assert item["total_bytes"] == 1000 and item["downloaded_bytes"] == 1000


async def test_failure_records_error(tmp_path):
    async def runner(repo_id, target, state, cancel):
        raise RuntimeError("network down")

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    await mgr._tasks["org/model"]
    [item] = mgr.snapshot()
    assert item["status"] == "error" and "network down" in item["error"]


async def test_cancel_interrupts_in_progress_download(tmp_path):
    started = asyncio.Event()

    async def runner(repo_id, target, state, cancel):
        started.set()
        await cancel.wait()  # a real runner would kill the subprocess here

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    task = mgr._tasks["org/model"]
    await started.wait()
    mgr.cancel("org/model")
    await task
    assert mgr.snapshot()[0]["status"] == "cancelled"


async def test_retry_after_error_succeeds(tmp_path):
    calls = {"n": 0}

    async def runner(repo_id, target, state, cancel):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("flaky")
        state.downloaded_bytes = 1000

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    await mgr._tasks["org/model"]
    assert mgr.snapshot()[0]["status"] == "error"
    mgr.retry("org/model")
    await mgr._tasks["org/model"]
    assert mgr.snapshot()[0]["status"] == "done"


async def test_state_persists_and_reloads(tmp_path):
    async def runner(repo_id, target, state, cancel):
        state.downloaded_bytes = 1000

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    await mgr._tasks["org/model"]
    # A fresh manager (simulating a restart) loads the persisted state.
    reborn = _manager(tmp_path, runner=runner)
    assert reborn.snapshot()[0] == {
        "repo_id": "org/model",
        "status": "done",
        "total_bytes": 1000,
        "downloaded_bytes": 1000,
        "eta_seconds": None,
        "error": None,
    }


async def test_resume_incomplete_respawns_interrupted_downloads(tmp_path):
    # Simulate an app that closed mid-download: a persisted "downloading" entry.
    (tmp_path / "downloads.json").write_text(
        json.dumps({"downloads": [{"repo_id": "org/model", "status": "downloading"}]})
    )
    resumed = asyncio.Event()

    async def runner(repo_id, target, state, cancel):
        resumed.set()
        state.downloaded_bytes = state.total_bytes

    mgr = _manager(tmp_path, runner=runner)
    await mgr.resume_incomplete()
    await mgr._tasks["org/model"]
    assert resumed.is_set() and mgr.snapshot()[0]["status"] == "done"


async def test_remove_drops_entry_and_does_not_resurrect(tmp_path):
    async def runner(repo_id, target, state, cancel):
        state.downloaded_bytes = 1000

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    await mgr._tasks["org/model"]
    assert len(mgr.snapshot()) == 1
    mgr.remove("org/model")
    assert mgr.snapshot() == []
    # a restart (fresh manager loading downloads.json) must not bring it back
    assert _manager(tmp_path, runner=runner).snapshot() == []


async def test_remove_unknown_raises(tmp_path):
    import pytest

    with pytest.raises(KeyError):
        _manager(tmp_path).remove("nope")


async def test_eta_reported_while_downloading(tmp_path):
    # A runner that pauses lets us inspect the in-flight public shape (eta from rate).
    hold = asyncio.Event()

    async def runner(repo_id, target, state, cancel):
        state.downloaded_bytes = 400
        state.rate_bps = 200.0  # 600 bytes left / 200 bps -> 3s
        await hold.wait()

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/model")
    task = mgr._tasks["org/model"]
    await asyncio.sleep(0.02)
    item = mgr.snapshot()[0]
    assert item["status"] == "downloading" and item["eta_seconds"] == 3
    hold.set()
    await task


def test_manager_binds_env_and_max_workers_onto_default_runner(tmp_path):
    # N50: with no injected runner, the manager binds the download tunables onto the real
    # subprocess runner (so the fixed Runner call site carries them through).
    import functools

    from assistant.downloads import _subprocess_runner

    mgr = DownloadManager(
        target_dir=tmp_path / "models",
        state_path=tmp_path / "downloads.json",
        env={"HF_HUB_DISABLE_XET": "1"},
        max_workers=2,
    )
    assert isinstance(mgr._runner, functools.partial)
    assert mgr._runner.func is _subprocess_runner
    assert mgr._runner.keywords == {"env": {"HF_HUB_DISABLE_XET": "1"}, "max_workers": 2}


async def test_subprocess_runner_passes_max_workers_and_merged_env(tmp_path, monkeypatch):
    # N50: the spawned command carries max_workers as an argv, and the child env merges the extra
    # hub tunables ONTO the parent env (PATH etc. must survive, not be replaced).
    import asyncio

    from assistant.downloads import DownloadState, _subprocess_runner

    captured: dict = {}

    class _FakeProc:
        returncode = 0
        pid = 999999
        stderr = None

        async def wait(self):
            return 0

    async def fake_exec(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    state = DownloadState(repo_id="org/m")
    await _subprocess_runner(
        "org/m", tmp_path / "t", state, asyncio.Event(),
        env={"HF_HUB_DISABLE_XET": "1"}, max_workers=3,
    )
    assert captured["args"][-1] == "3"  # max_workers is the trailing argv
    assert captured["args"][3] == "org/m"  # repo_id positional preserved
    assert captured["env"]["HF_HUB_DISABLE_XET"] == "1"  # extra tunable applied
    assert "PATH" in captured["env"]  # merged onto parent env, not a bare replacement


async def test_downloads_run_one_at_a_time(tmp_path):
    # N52: model files are large, so only ONE download transfers at a time; others wait "queued".
    release = asyncio.Event()
    started: list[str] = []

    async def runner(repo_id, target, state, cancel):
        started.append(repo_id)
        if repo_id == "org/first":
            await release.wait()  # hold the single slot open

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/first")
    mgr.start("org/second")
    t1, t2 = mgr._tasks["org/first"], mgr._tasks["org/second"]
    await asyncio.sleep(0.05)  # let both tasks reach the gate

    snap = {d["repo_id"]: d["status"] for d in mgr.snapshot()}
    assert snap["org/first"] == "downloading"
    assert snap["org/second"] == "queued"  # waiting its turn
    assert started == ["org/first"]  # the second's runner has NOT started yet

    release.set()  # first finishes → the queued one proceeds
    await asyncio.gather(t1, t2)
    assert started == ["org/first", "org/second"]  # ran sequentially
    assert all(d["status"] == "done" for d in mgr.snapshot())


async def test_cancel_a_queued_download_takes_effect_immediately(tmp_path):
    # A queued download can be cancelled without waiting for the active one to finish.
    release = asyncio.Event()

    async def runner(repo_id, target, state, cancel):
        if repo_id == "org/first":
            await release.wait()

    mgr = _manager(tmp_path, runner=runner)
    mgr.start("org/first")
    mgr.start("org/second")
    t1 = mgr._tasks["org/first"]
    await asyncio.sleep(0.05)
    assert {d["repo_id"]: d["status"] for d in mgr.snapshot()}["org/second"] == "queued"

    mgr.cancel("org/second")  # cancel while still queued
    await asyncio.sleep(0.02)
    statuses = {d["repo_id"]: d["status"] for d in mgr.snapshot()}
    assert statuses["org/second"] == "cancelled"  # took effect without waiting its turn
    assert statuses["org/first"] == "downloading"  # the active download is unaffected

    release.set()
    await t1
    assert {d["repo_id"]: d["status"] for d in mgr.snapshot()}["org/first"] == "done"
