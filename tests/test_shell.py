import pytest

from assistant.tools import build_registry
from assistant.tools.base import ToolContext


@pytest.fixture
def bash():
    return build_registry().get("bash")


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
