"""Reasoning-knob detection off a checkpoint's chat template.

The stakes are asymmetric: an unsupported ``enable_thinking`` is ignored by jinja, but an
``reasoning_effort`` value a template doesn't accept can raise inside the render and kill the
turn — so the value list has to come from the model, never from a hardcoded menu.
"""

from __future__ import annotations

import json

from assistant.models.chat_template import reasoning_support, template_source

# Abridged from mlx-community/Qwen3.8-27B-8bit: validates the effort against its own set (note
# it has no "high") and raises otherwise — the exact shape the picker must read.
QWEN38 = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}
    {%- endif %}
{%- endif %}
"""

# Abridged from gpt-oss's harmony template: interpolates the effort without validating it, so
# there is no list to read.
GPT_OSS = """
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort + "\\n" }}
"""

PLAIN = "{% for m in messages %}{{ m.content }}{% endfor %}"


def test_enumerated_effort_values_come_from_the_template(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(QWEN38)
    caps = reasoning_support(tmp_path)
    assert caps["thinking"] is True
    # Ordered weakest→strongest for the picker, and "high" is absent because Qwen3.8 rejects it.
    assert caps["effort"] == ["low", "medium", "xhigh"]


def test_unvalidated_effort_falls_back_to_the_standard_triple(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(GPT_OSS)
    caps = reasoning_support(tmp_path)
    assert caps["effort"] == ["low", "medium", "high"]
    assert caps["thinking"] is False  # harmony has no enable_thinking — the toggle must hide


def test_template_without_reasoning_knobs_offers_none(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(PLAIN)
    assert reasoning_support(tmp_path) == {"thinking": False, "effort": []}


def test_missing_template_is_not_an_error(tmp_path):
    assert template_source(tmp_path) is None
    assert reasoning_support(tmp_path) == {"thinking": False, "effort": []}


def test_template_embedded_in_tokenizer_config_is_found(tmp_path):
    # Older checkpoints ship no chat_template.jinja; the template lives in tokenizer_config.json.
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN38}))
    assert reasoning_support(tmp_path)["effort"] == ["low", "medium", "xhigh"]


def test_multi_template_checkpoints_are_scanned_whole(tmp_path):
    # {"name": …, "template": …} entries: the knob may live in any one of them.
    (tmp_path / "chat_template.json").write_text(
        json.dumps({"chat_template": [{"name": "tool_use", "template": QWEN38},
                                      {"name": "default", "template": PLAIN}]})
    )
    assert reasoning_support(tmp_path)["thinking"] is True


def test_standalone_jinja_wins_over_tokenizer_config(tmp_path):
    # Preferring the small file is a cost decision (tokenizer_config.json carries the vocabulary),
    # so it must actually be the one that answers when both exist.
    (tmp_path / "chat_template.jinja").write_text(PLAIN)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN38}))
    assert reasoning_support(tmp_path) == {"thinking": False, "effort": []}
