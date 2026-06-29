"""Backend-authoritative default model store (Spring 3 S3.2c)."""

from __future__ import annotations

from assistant.models.default_store import DefaultModelStore


def test_seed_used_when_no_file(tmp_path):
    assert DefaultModelStore(tmp_path / "d.json", seed="a").value == "a"


def test_set_persists_and_reloads_over_seed(tmp_path):
    p = tmp_path / "d.json"
    DefaultModelStore(p, seed="a").set("b")
    # A fresh store reads the persisted value, not the seed.
    assert DefaultModelStore(p, seed="a").value == "b"


def test_set_none_clears(tmp_path):
    s = DefaultModelStore(tmp_path / "d.json", seed="a")
    s.set(None)
    assert s.value is None
