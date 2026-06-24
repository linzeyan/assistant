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
