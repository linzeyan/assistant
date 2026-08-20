"""Which reasoning knobs a model's chat template actually understands.

Read straight off the checkpoint's own template source (no weights, no tokenizer load), because
the two knobs behave very differently when a model doesn't support them:

- ``enable_thinking`` is *safe to send blind* — jinja ignores variables a template never reads.
- ``reasoning_effort`` is NOT. Qwen3.8's template validates it and ``raise_exception``s on an
  unknown value (its set is ``low``/``medium``/``xhigh`` — note: no ``high``), which surfaces as a
  failed render and kills the whole turn. So the value list has to come from the model itself.

The result drives the pickers: hide a knob the model can't use, and offer only values it accepts.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

# Weakest → strongest, so the picker reads in a sensible order no matter what order the template
# happens to list its values in. Values outside this table keep their template order, after these.
_EFFORT_RANK = ("minimal", "low", "medium", "high", "xhigh")

# The membership test a template validates the effort against, e.g. Qwen3.8's
#   {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
# Anchored on the variable *name*, so a renamed local (``resolved_reasoning_effort``) still hits.
_EFFORT_CHOICES_RE = re.compile(r"reasoning_effort\s+(?:not\s+)?in\s*[\(\[]([^)\]]*)[)\]]")
_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")

# Templates that interpolate the effort without validating it (gpt-oss's harmony template just
# writes ``Reasoning: {{ reasoning_effort }}``) publish no value list. Offering the de-facto
# standard triple beats hiding a knob the model does honour — and an unvalidated template can't
# raise on a wrong guess, which is exactly why the guess is safe here and nowhere else.
_EFFORT_FALLBACK = ("low", "medium", "high")


def template_source(model_dir: Path) -> str | None:
    """The model's chat template as jinja source, or None when it ships none.

    Checks the standalone ``chat_template.jinja`` first: it's a few KB, while the
    ``tokenizer_config.json`` fallback can be tens of MB of vocabulary we'd parse for one key.
    """
    jinja = model_dir / "chat_template.jinja"
    if jinja.is_file():
        try:
            return jinja.read_text()
        except OSError:
            return None
    for name in ("tokenizer_config.json", "chat_template.json"):
        f = model_dir / name
        if not f.is_file():
            continue
        try:
            raw = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        tpl = raw.get("chat_template") if isinstance(raw, dict) else None
        if isinstance(tpl, str):
            return tpl
        if isinstance(tpl, list):  # multi-template checkpoints: {"name":…, "template":…} entries
            parts = [t.get("template", "") for t in tpl if isinstance(t, dict)]
            if parts:
                return "\n".join(parts)
    return None


def _effort_values(text: str) -> list[str]:
    if "reasoning_effort" not in text:
        return []
    values: list[str] = []
    for clause in _EFFORT_CHOICES_RE.findall(text):
        for v in _QUOTED_RE.findall(clause):
            if v not in values:
                values.append(v)
    if not values:
        return list(_EFFORT_FALLBACK)
    return sorted(values, key=lambda v: (_EFFORT_RANK.index(v) if v in _EFFORT_RANK else len(_EFFORT_RANK)))


def reasoning_support(model_dir: Path) -> dict:
    """``{"thinking": bool, "effort": [values]}`` for this checkpoint.

    ``effort == []`` means the template never reads ``reasoning_effort``, so the knob must not be
    offered at all — sending it would be a no-op at best and a render exception at worst.
    """
    text = template_source(model_dir) or ""
    return {"thinking": "enable_thinking" in text, "effort": _effort_values(text)}
