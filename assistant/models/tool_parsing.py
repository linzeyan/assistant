"""Parse tool calls out of a local model's completed generation.

Unlike omlx (which speaks OpenAI's wire format and streams structured
``delta.tool_calls``), mlx-lm emits tool calls as *text* in a model-specific
format. This module turns that text back into the same structured shape the agent
loop already consumes, so the native backend reaches full tool-calling parity.

Supported formats (the common ones across MLX-community tool-capable models):

* Hermes / Qwen — ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks
* Mistral       — ``[TOOL_CALLS][{"name": ..., "arguments": {...}}, ...]``
* Llama 3.1     — ``<|python_tag|>{"name": ..., "parameters": {...}}``
* Qwen XML      — ``<tool_call><function=NAME><parameter=KEY>VALUE</parameter>…</function></tool_call>``
                  nested-XML carrying *no JSON*, as some Qwen3.x templates emit (notably
                  via mlx-vlm). Parsed only as a fallback when the block holds no JSON.
* Bare JSON     — a lone ``{...}``/``[...]`` describing a call (Llama without the
                  tag). Ambiguous, so only accepted when its name is a known tool.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_PYTHON_TAG = "<|python_tag|>"
_MISTRAL_TAG = "[TOOL_CALLS]"
_HERMES_OPEN = "<tool_call>"

# Qwen3.x nested-XML tool calls (no JSON), e.g. emitted by some mlx-vlm templates:
#   <function=web_search><parameter=query>…</parameter></function>
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

# Markers a streaming consumer watches for to stop emitting text and start
# buffering a tool call. Bare JSON has no marker (handled by a leading-brace check).
TOOL_MARKERS = (_HERMES_OPEN, _PYTHON_TAG, _MISTRAL_TAG)


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: dict


def _iter_json_values(s: str) -> list:
    """Pull every top-level JSON object/array out of ``s``, ignoring surrounding
    prose. Tolerant of multiple values and junk between them."""
    decoder = json.JSONDecoder()
    out: list = []
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] not in "{[":
            i += 1
        if i >= n:
            break
        try:
            obj, end = decoder.raw_decode(s, i)
        except json.JSONDecodeError:
            i += 1
            continue
        out.append(obj)
        i = end
    return out


def _coerce_call(obj) -> ParsedToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    # Models disagree on the args key ("arguments" vs Llama's "parameters").
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters", {})
    if isinstance(args, str):
        # Some templates double-encode arguments as a JSON string.
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"__raw_arguments__": args}
    if not isinstance(args, dict):
        args = {"value": args}
    return ParsedToolCall(id="", name=name, arguments=args)


def _coerce_all(values: list, restrict: set[str] | None) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            call = _coerce_call(item)
            if call is None:
                continue
            if restrict is not None and call.name not in restrict:
                continue
            calls.append(call)
    for index, call in enumerate(calls):
        call.id = f"call_{index}"
    return calls


def _parse_xml_functions(text: str) -> list[ParsedToolCall]:
    """Parse the nested-XML tool-call form some templates emit instead of JSON.

    Qwen3.x (notably via mlx-vlm) renders a call as
    ``<function=web_search><parameter=query>台北今天天氣</parameter></function>`` — the
    name in the tag, one ``<parameter=KEY>VALUE</parameter>`` per argument, never JSON.
    ``_iter_json_values`` finds nothing here, so without this branch the block leaks back
    to the user as raw text and the tool is never called.
    """
    calls: list[ParsedToolCall] = []
    for fn in _FUNCTION_RE.finditer(text):
        name = fn.group(1).strip()
        if not name:
            continue
        args: dict = {}
        for param in _PARAM_RE.finditer(fn.group(2)):
            raw = param.group(2).strip()
            try:
                # Recover real types (numbers, bools, objects) when the value is JSON;
                # otherwise keep the plain string (the common case, e.g. a search query).
                value = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                value = raw
            args[param.group(1).strip()] = value
        calls.append(ParsedToolCall(id="", name=name, arguments=args))
    for index, call in enumerate(calls):
        call.id = f"call_{index}"
    return calls


def parse_tool_calls(
    text: str, known_names: set[str] | None = None
) -> list[ParsedToolCall]:
    """Extract tool calls from ``text``. Returns ``[]`` when there are none.

    ``known_names`` gates the *bare JSON* fallback only: a lone JSON object is too
    easily confused with a legitimate JSON answer, so it's treated as a tool call
    only when its ``name`` matches a known tool. Explicit marker formats are trusted
    regardless (an unknown name is surfaced cleanly by the agent loop's dispatch).
    """
    blocks = _TOOL_CALL_RE.findall(text)
    if blocks:
        values: list = []
        for block in blocks:
            values.extend(_iter_json_values(block))
        calls = _coerce_all(values, restrict=None)
        if calls:
            return calls
        # No JSON inside the <tool_call> block: fall back to the nested-XML function
        # form (Qwen3.x, commonly via mlx-vlm), which carries no JSON at all. Without
        # this the block would leak back as raw text and the tool would never run.
        return _parse_xml_functions("\n".join(blocks))

    if _HERMES_OPEN in text:  # opened but never closed (truncated generation)
        after = text.split(_HERMES_OPEN, 1)[1]
        return _coerce_all(_iter_json_values(after), restrict=None)

    if _MISTRAL_TAG in text:
        after = text.split(_MISTRAL_TAG, 1)[1]
        return _coerce_all(_iter_json_values(after), restrict=None)

    if _PYTHON_TAG in text:
        after = text.split(_PYTHON_TAG, 1)[1]
        return _coerce_all(_iter_json_values(after), restrict=None)

    stripped = text.strip()
    if stripped[:1] in "{[" and known_names:
        return _coerce_all(_iter_json_values(stripped), restrict=known_names)
    return []


def earliest_marker(buffer: str, start: int = 0) -> int | None:
    """Index of the first tool-call marker at/after ``start``, or ``None``."""
    best: int | None = None
    for marker in TOOL_MARKERS:
        idx = buffer.find(marker, start)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best
