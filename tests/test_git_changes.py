"""git-based discovery of shell-touched files for the turn diff (Spring 2 P3)."""

from __future__ import annotations

import subprocess

from assistant.agent.git_changes import dirty_paths, head_bytes, repo_root


def _init_repo(d):
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.name", "t"], check=True)
    (d / "a.txt").write_text("hello\n")
    subprocess.run(["git", "-C", str(d), "add", "."], check=True)
    subprocess.run(["git", "-C", str(d), "commit", "-qm", "init"], check=True)


def test_repo_root_is_none_outside_a_repo(tmp_path):
    assert repo_root(tmp_path) is None


def test_dirty_paths_lists_modified_and_untracked(tmp_path):
    _init_repo(tmp_path)
    root = repo_root(tmp_path)
    assert root is not None
    (tmp_path / "a.txt").write_text("changed\n")
    (tmp_path / "b.txt").write_text("new\n")
    dirty = dirty_paths(root)
    assert "a.txt" in dirty and "b.txt" in dirty


def test_head_bytes_returns_committed_or_none(tmp_path):
    _init_repo(tmp_path)
    root = repo_root(tmp_path)
    assert head_bytes(root, "a.txt") == b"hello\n"  # committed content
    assert head_bytes(root, "never.txt") is None  # not tracked
