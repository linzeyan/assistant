"""Unified tool-output bounding (S4). The point: keep BOTH ends (a head-only cap dropped the
tail where errors/results live) and preserve the full output via spill so nothing is lost."""

from assistant.tools.output_bounding import bound_text


def test_short_text_unchanged():
    assert bound_text("hello", limit=100) == "hello"


def test_keeps_both_ends_and_elides_middle():
    text = "H" * 1000 + "M" * 1000 + "T" * 1000
    out = bound_text(text, limit=1000)
    assert out.startswith("H")  # head kept
    assert out.rstrip().endswith("T")  # tail kept — the part a head-only cap would drop
    assert "omitted" in out
    assert "M" * 100 not in out  # middle elided
    assert len(out) < 1500  # bounded to ~limit + marker, not the full 3000


def test_spill_writes_full_text_and_names_the_path(tmp_path):
    text = "x" * 5000
    out = bound_text(text, limit=1000, spill_dir=tmp_path, label="bash")
    assert "full 5000 chars at" in out
    spilled = list(tmp_path.glob("bash-*.txt"))
    assert len(spilled) == 1 and spilled[0].read_text() == text


def test_spill_is_content_addressed_and_idempotent(tmp_path):
    text = "y" * 5000
    bound_text(text, limit=1000, spill_dir=tmp_path, label="bash")
    bound_text(text, limit=1000, spill_dir=tmp_path, label="bash")
    assert len(list(tmp_path.glob("bash-*.txt"))) == 1  # identical content -> one file


def test_spill_failure_still_bounds(tmp_path):
    blocker = tmp_path / "afile"
    blocker.write_text("x")  # a file where a dir is expected -> mkdir fails -> no spill
    out = bound_text("z" * 5000, limit=1000, spill_dir=blocker / "sub", label="bash")
    assert "omitted" in out and "full" not in out  # bounded, just no spill pointer


def test_gc_spill_dir_prunes_only_old_files(tmp_path):
    # S15/H8: GC removes spill files past the retention age and leaves fresh ones, returning the
    # count + bytes freed. mtime is backdated rather than waiting on the clock.
    import os
    import time

    from assistant.tools.output_bounding import gc_spill_dir

    old = tmp_path / "bash-aaaaaaaaaaaa.txt"
    fresh = tmp_path / "bash-bbbbbbbbbbbb.txt"
    old.write_text("x" * 100)
    fresh.write_text("y" * 10)
    old_time = time.time() - 30 * 86_400  # 30 days ago
    os.utime(old, (old_time, old_time))

    removed, freed = gc_spill_dir(tmp_path, max_age_days=14)
    assert removed == 1 and freed == 100
    assert not old.exists() and fresh.exists()


def test_gc_spill_dir_disabled_and_missing_dir(tmp_path):
    from assistant.tools.output_bounding import gc_spill_dir

    f = tmp_path / "bash-cccccccccccc.txt"
    f.write_text("z")
    import os
    import time

    ago = time.time() - 999 * 86_400
    os.utime(f, (ago, ago))
    assert gc_spill_dir(tmp_path, max_age_days=0) == (0, 0)  # 0 disables → nothing removed
    assert f.exists()
    assert gc_spill_dir(tmp_path / "nope", max_age_days=14) == (0, 0)  # missing dir → no-op
