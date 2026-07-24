"""Per-model generation overrides store + the mlx sampler helper (Spring 3 S3.2b)."""

from __future__ import annotations

import pytest

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


def test_type_override_is_stored_and_validated(tmp_path):
    s = PerModelStore(tmp_path / "p.json")
    s.set("m", {"type": "llm"})
    assert s.kind_override("m") == "llm"
    # An unrecognised / "auto" type clears the override (falls back to auto-detection).
    s.set("m", {"type": "auto"})
    assert s.kind_override("m") is None
    s.set("m", {"type": "bogus"})
    assert s.kind_override("m") is None


def test_type_override_is_kept_out_of_generation_params(tmp_path):
    # The type override must NOT leak into sampling kwargs — a stray `type` would break generation.
    s = PerModelStore(tmp_path / "p.json")
    s.set("m", {"temperature": 0.5, "type": "vlm"})
    assert s.generation("m") == {"temperature": 0.5}  # type excluded
    assert s.kind_override("m") == "vlm"
    assert s.get("m") == {"temperature": 0.5, "type": "vlm"}  # full view keeps both


def test_chat_template_kwargs_stored_validated_and_isolated(tmp_path):
    # 2-B: per-model chat-template variables (e.g. Qwen3.x enable_thinking). Stored as a dict of
    # scalars, surfaced via its own getter, and NEVER merged into generation params — a stray
    # dict kwarg in sampler args would break generation.
    s = PerModelStore(tmp_path / "p.json")
    out = s.set("m", {"chat_template_kwargs": {"enable_thinking": False, "custom_var": "x"}})
    assert out["chat_template_kwargs"] == {"enable_thinking": False, "custom_var": "x"}
    assert s.chat_template_kwargs("m") == {"enable_thinking": False, "custom_var": "x"}
    assert s.generation("m") == {}  # template kwargs stay out of sampler params
    assert s.chat_template_kwargs("unset") == {}
    # Persists across reload alongside the other concerns.
    assert PerModelStore(tmp_path / "p.json").chat_template_kwargs("m")["enable_thinking"] is False


def test_chat_template_kwargs_rejects_junk(tmp_path):
    s = PerModelStore(tmp_path / "p.json")
    # Non-dict never stores; non-scalar values / non-str keys are dropped (jinja vars are scalars).
    s.set("m", {"chat_template_kwargs": "not a dict", "temperature": 0.5})
    assert s.chat_template_kwargs("m") == {}
    s.set("m", {"chat_template_kwargs": {"ok": True, "nested": {"no": 1}, 3: "bad-key"}})
    assert s.chat_template_kwargs("m") == {"ok": True}
    # An all-junk dict clears the entry entirely.
    s.set("m", {"chat_template_kwargs": {"nested": [1, 2]}})
    assert "chat_template_kwargs" not in s.get("m")


def test_sampler_kwargs_empty_when_no_overrides():
    # No overrides → no sampler kwarg (mlx-lm keeps its own default), and crucially no mlx-lm
    # import is forced, so the helper is safe to call on a machine without mlx installed.
    assert _sampler_kwargs(None, None, None) == {}


def test_sampler_kwargs_builds_sampler_when_set():
    # Needs a REAL mlx_lm — the point is that the kwargs wire into make_sampler. On a
    # machine without a working mlx stack (CI installs dev deps only; a source venv can
    # hold a transformers/huggingface-hub pin conflict that breaks the import chain),
    # skip visibly instead of failing forever.
    pytest.importorskip("mlx_lm", exc_type=ImportError)
    out = _sampler_kwargs(0.7, 0.9, 40)
    assert "sampler" in out and callable(out["sampler"])
