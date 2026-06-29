"""Per-model generation overrides store + the mlx sampler helper (Spring 3 S3.2b)."""

from __future__ import annotations

from assistant.models.mlx_engine import _sampler_kwargs
from assistant.models.per_model_store import PerModelStore


def test_set_get_filters_unknown_and_null(tmp_path):
    s = PerModelStore(tmp_path / "p.json")
    out = s.set("m", {"temperature": 0.7, "top_k": 40, "bogus": 1, "top_p": None})
    assert out == {"temperature": 0.7, "top_k": 40}  # unknown + null dropped
    assert s.get("m") == {"temperature": 0.7, "top_k": 40}
    assert s.get("other") == {}  # unset model → empty


def test_persists_and_reloads(tmp_path):
    p = tmp_path / "p.json"
    PerModelStore(p).set("m", {"temperature": 0.3})
    assert PerModelStore(p).get("m") == {"temperature": 0.3}


def test_empty_update_clears_overrides(tmp_path):
    p = tmp_path / "p.json"
    s = PerModelStore(p)
    s.set("m", {"temperature": 0.3})
    s.set("m", {"temperature": None})  # all-null → clear
    assert s.get("m") == {}
    assert PerModelStore(p).get("m") == {}  # cleared on disk too


def test_sampler_kwargs_empty_when_no_overrides():
    # No overrides → no sampler kwarg (mlx-lm keeps its own default), and crucially no mlx-lm
    # import is forced, so the helper is safe to call on a machine without mlx installed.
    assert _sampler_kwargs(None, None, None) == {}


def test_sampler_kwargs_builds_sampler_when_set():
    out = _sampler_kwargs(0.7, 0.9, 40)
    assert "sampler" in out and callable(out["sampler"])
