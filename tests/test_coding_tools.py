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


async def test_a_failed_anchor_says_where_the_candidates_are(ctx, tools):
    """Both failures name line numbers, because the observed failure mode is a model that
    resends the identical anchor until the loop stops it. The anchor is nearly always right
    except for indentation it dropped — which is also what makes a unique anchor ambiguous —
    so the lines that nearly match are the one thing that lets it fix the call itself."""
    (ctx.cwd / "s.swift").write_text("func run() {\n    check()\n}\n\nfunc check() {\n}\n")

    dup = await tools["edit_file"].handler(
        {"path": "s.swift", "old": "check()", "new": "check()\n    other()"}, ctx
    )
    assert dup.ok is False and "not unique" in dup.content
    assert "2: " in dup.content, dup.content

    # The anchor a model actually sends when it fails: the first line is right and the
    # indentation of the second is not, so the whole snippet matches nothing.
    missing = await tools["edit_file"].handler(
        {"path": "s.swift", "old": "func run() {\ncheck()", "new": "x"}, ctx
    )
    assert missing.ok is False and "not found" in missing.content
    assert "1: func run() {" in missing.content, missing.content


async def test_glob_and_grep(ctx, tools):
    (ctx.cwd / "one.py").write_text("import os\nVALUE = 1\n")
    (ctx.cwd / "two.py").write_text("VALUE = 2\n")
    g = await tools["glob"].handler({"pattern": "*.py"}, ctx)
    assert "one.py" in g.content and "two.py" in g.content

    hits = await tools["grep"].handler({"pattern": r"VALUE = \d"}, ctx)
    assert hits.ok and hits.content.count("VALUE =") == 2


async def test_grep_survives_an_unstattable_entry(ctx, tools, monkeypatch):
    # Real failure from an A1 sweep: is_file() RAISES (not returns False) on paths this process
    # can't stat — macOS system paths, other users' dirs. One such entry under the root aborted
    # the entire search, so the caller got an error instead of every other file's matches.
    from pathlib import Path

    (ctx.cwd / "ok.txt").write_text("match here\n")
    (ctx.cwd / "vault").write_text("")
    real_is_file = Path.is_file

    def exploding_is_file(self):
        if self.name == "vault":
            raise PermissionError(13, "Permission denied")
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", exploding_is_file)
    hits = await tools["grep"].handler({"pattern": "match"}, ctx)
    assert hits.ok, hits.content  # the unreadable entry must not fail the search
    assert "ok.txt" in hits.content  # and the readable file's match still comes back


async def test_grep_skips_binary(ctx, tools):
    (ctx.cwd / "bin.dat").write_bytes(b"\x00\x01\x02match\xff")
    (ctx.cwd / "ok.txt").write_text("match here\n")
    hits = await tools["grep"].handler({"pattern": "match"}, ctx)
    # Only the text file should appear; the binary file is skipped, not crashed on.
    assert "ok.txt" in hits.content and "bin.dat" not in hits.content


async def test_a_write_outside_the_workspace_is_refused(ctx, tools, tmp_path):
    """The standing approval rule pre-authorises edit_file and write_file, so this
    boundary is the only thing left between "edit a file in the project" and "edit
    any file on the machine". A resource glob cannot express it: the rule sees the
    path as the model wrote it, and ../ reaches the same file under another name."""
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("untouched")

    res = await tools["write_file"].handler({"path": str(outside), "content": "x"}, ctx)
    assert not res.ok and "outside the workspace" in res.content
    assert outside.read_text() == "untouched"

    res = await tools["write_file"].handler({"path": "../outside.txt", "content": "x"}, ctx)
    assert not res.ok and "outside the workspace" in res.content
    assert outside.read_text() == "untouched"

    res = await tools["edit_file"].handler(
        {"path": "../outside.txt", "old": "untouched", "new": "x"}, ctx
    )
    assert not res.ok and "outside the workspace" in res.content
    assert outside.read_text() == "untouched"


async def test_a_symlink_out_of_the_workspace_is_refused(ctx, tools, tmp_path):
    """Resolved rather than compared as text, or a link inside the tree would be a
    door out of it."""
    outside = tmp_path.parent / "linked.txt"
    outside.write_text("untouched")
    (ctx.cwd / "link.txt").symlink_to(outside)

    res = await tools["edit_file"].handler(
        {"path": "link.txt", "old": "untouched", "new": "x"}, ctx
    )
    assert not res.ok and "outside the workspace" in res.content
    assert outside.read_text() == "untouched"


async def test_reading_outside_the_workspace_is_still_allowed(ctx, tools, tmp_path):
    """Only the mutating tools are bounded. An agent asked about a sibling checkout
    or a config file is doing something ordinary, and a read damages nothing."""
    outside = tmp_path.parent / "readable.txt"
    outside.write_text("visible")
    res = await tools["read_file"].handler({"path": str(outside)}, ctx)
    assert res.ok and "visible" in res.content
