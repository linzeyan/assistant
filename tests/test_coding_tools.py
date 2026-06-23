import pytest

from assistant.tools import build_registry
from assistant.tools.base import ToolContext


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=tmp_path)


@pytest.fixture
def tools():
    reg = build_registry()
    return {n: reg.get(n) for n in ("read_file", "write_file", "edit_file", "glob", "grep")}


async def test_write_then_read_roundtrip(ctx, tools):
    res = await tools["write_file"].handler({"path": "a.txt", "content": "hello"}, ctx)
    assert res.ok and (ctx.cwd / "a.txt").read_text() == "hello"
    read = await tools["read_file"].handler({"path": "a.txt"}, ctx)
    assert read.ok and read.content == "hello"


async def test_read_missing_file_fails_soft(ctx, tools):
    res = await tools["read_file"].handler({"path": "nope.txt"}, ctx)
    assert res.ok is False and "not a file" in res.content


async def test_edit_requires_unique_match(ctx, tools):
    (ctx.cwd / "f.txt").write_text("x x")
    dup = await tools["edit_file"].handler({"path": "f.txt", "old": "x", "new": "y"}, ctx)
    assert dup.ok is False and "not unique" in dup.content

    (ctx.cwd / "g.txt").write_text("alpha beta")
    ok = await tools["edit_file"].handler({"path": "g.txt", "old": "beta", "new": "gamma"}, ctx)
    assert ok.ok and (ctx.cwd / "g.txt").read_text() == "alpha gamma"


async def test_glob_and_grep(ctx, tools):
    (ctx.cwd / "one.py").write_text("import os\nVALUE = 1\n")
    (ctx.cwd / "two.py").write_text("VALUE = 2\n")
    g = await tools["glob"].handler({"pattern": "*.py"}, ctx)
    assert "one.py" in g.content and "two.py" in g.content

    hits = await tools["grep"].handler({"pattern": r"VALUE = \d"}, ctx)
    assert hits.ok and hits.content.count("VALUE =") == 2


async def test_grep_skips_binary(ctx, tools):
    (ctx.cwd / "bin.dat").write_bytes(b"\x00\x01\x02match\xff")
    (ctx.cwd / "ok.txt").write_text("match here\n")
    hits = await tools["grep"].handler({"pattern": "match"}, ctx)
    # Only the text file should appear; the binary file is skipped, not crashed on.
    assert "ok.txt" in hits.content and "bin.dat" not in hits.content
