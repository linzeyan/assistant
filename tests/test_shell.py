import os

import pytest

from assistant.tools import build_registry, shell
from assistant.tools.base import ToolContext


@pytest.fixture
def bash():
    return build_registry().get("bash")


@pytest.fixture(autouse=True)
def _path_cache():
    """Pin the process-wide PATH cache so no test spawns the developer's login shell — that
    would make the suite depend on whatever their profile happens to print, and pay for it.
    ``None`` is the "resolution failed, keep what we inherited" value; the tests that are
    about resolution itself reset this to the sentinel first."""
    shell._user_path_cache = None
    yield
    shell._user_path_cache = None


async def test_bash_runs_with_the_login_shells_path(tmp_path, bash, monkeypatch):
    """Started by the .app, the backend inherits launchd's four-entry PATH, so the user's
    toolchain (cargo, rg, brew) is invisible and every command that needs it exits 127."""
    monkeypatch.setattr(shell, "_user_path_cache", "/opt/toolchain/bin:/usr/bin:/bin")
    res = await bash.handler({"command": "echo $PATH"}, ToolContext(cwd=tmp_path))
    assert "/opt/toolchain/bin" in res.content


async def test_bash_keeps_the_inherited_path_when_resolution_fails(tmp_path, bash, monkeypatch):
    monkeypatch.setattr(shell, "_user_path_cache", None)
    res = await bash.handler({"command": "echo $PATH"}, ToolContext(cwd=tmp_path))
    assert os.environ["PATH"] in res.content


async def test_user_path_is_resolved_once(monkeypatch):
    calls = []

    async def fake_exec(*args, **kwargs):
        calls.append(args)

        class _Proc:
            async def communicate(self):
                return (b"/a/bin:/b/bin", b"")

        return _Proc()

    monkeypatch.setattr(shell, "_user_path_cache", shell._UNRESOLVED)
    monkeypatch.setattr(shell.asyncio, "create_subprocess_exec", fake_exec)
    assert await shell._user_path() == "/a/bin:/b/bin"
    assert await shell._user_path() == "/a/bin:/b/bin"
    assert len(calls) == 1  # the login shell is expensive; asking it twice is a bug


async def test_user_path_rejects_a_single_entry_answer(monkeypatch):
    """A profile that prints a banner, or a shell that fails silently, can yield one word.
    Treating that as PATH would be worse than keeping what we inherited."""

    async def fake_exec(*args, **kwargs):
        class _Proc:
            async def communicate(self):
                return (b"welcome to your shell", b"")

        return _Proc()

    monkeypatch.setattr(shell, "_user_path_cache", shell._UNRESOLVED)
    monkeypatch.setattr(shell.asyncio, "create_subprocess_exec", fake_exec)
    assert await shell._user_path() is None


async def test_bash_success(tmp_path, bash):
    res = await bash.handler({"command": "echo hello"}, ToolContext(cwd=tmp_path))
    assert res.ok and "hello" in res.content and "[exit 0]" in res.content


async def test_bash_nonzero_exit_is_not_ok(tmp_path, bash):
    res = await bash.handler({"command": "exit 3"}, ToolContext(cwd=tmp_path))
    assert res.ok is False and "[exit 3]" in res.content


async def test_bash_respects_cwd(tmp_path, bash):
    (tmp_path / "marker.txt").write_text("x")
    res = await bash.handler({"command": "ls"}, ToolContext(cwd=tmp_path))
    assert "marker.txt" in res.content


async def test_bash_times_out(tmp_path, bash):
    res = await bash.handler(
        {"command": "sleep 5", "timeout": 0.5}, ToolContext(cwd=tmp_path)
    )
    assert res.ok is False and "timed out" in res.content
