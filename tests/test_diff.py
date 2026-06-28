"""Unified-diff builder for turn changes (Spring 2 P2/P3 code-result return)."""

from __future__ import annotations

from assistant.agent.diff import build_turn_changes


def test_added_file_counts_all_lines_as_additions():
    c = build_turn_changes({"new.py": (None, b"a\nb\nc\n")})
    assert len(c.files) == 1
    f = c.files[0]
    assert (f.path, f.status, f.additions, f.deletions) == ("new.py", "added", 3, 0)
    assert "+a" in c.diff and c.summary() == "1 file changed (+3/-0)"


def test_modified_file_counts_adds_and_dels():
    c = build_turn_changes({"x.py": (b"a\nb\n", b"a\nc\n")})
    f = c.files[0]
    assert f.status == "modified" and f.additions == 1 and f.deletions == 1


def test_deleted_file():
    c = build_turn_changes({"gone.py": (b"x\n", None)})
    assert c.files[0].status == "deleted" and c.files[0].deletions == 1


def test_net_unchanged_is_skipped():
    # An edit that wrote identical bytes is not a change — it must not appear at all.
    c = build_turn_changes({"same.py": (b"a\n", b"a\n")})
    assert c.files == [] and c.diff == ""


def test_binary_recorded_but_not_diffed():
    c = build_turn_changes({"img.bin": (b"\x00\x01", b"\x00\x02")})
    assert c.files[0].status == "modified" and c.files[0].additions == 0
    assert "binary" in c.diff and "\x00" not in c.diff


def test_diff_is_truncated_when_huge():
    big_after = ("x\n" * 200_000).encode()
    c = build_turn_changes({"big.txt": (None, big_after)})
    assert len(c.diff) < 200_000 and c.diff.endswith("(diff truncated)")


def test_multiple_files_sorted_and_summed():
    c = build_turn_changes(
        {"b.py": (None, b"1\n"), "a.py": (b"x\n", b"y\n")}
    )
    assert [f.path for f in c.files] == ["a.py", "b.py"]  # sorted
    assert c.summary() == "2 files changed (+2/-1)"
