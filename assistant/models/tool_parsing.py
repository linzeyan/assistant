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
import uuid
from dataclasses import dataclass

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_PYTHON_TAG = "<|python_tag|>"
_MISTRAL_TAG = "[TOOL_CALLS]"
_HERMES_OPEN = "<tool_call>"

# Qwen3.x nested-XML tool calls (no JSON), e.g. emitted by some mlx-vlm templates:
#   <function=web_search><parameter=query>…</parameter></function>
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)

# Harmony (OpenAI gpt-oss): a tool call rides the "commentary" channel as
#   ...to=functions.NAME <|constrain|>json<|message|>{...json args...}<|call|>
# The recipient carries the tool name; the JSON payload follows <|message|> up to <|call|>.
# UNVERIFIED against real gpt-oss output — the model isn't downloaded yet, and how mlx-lm's
# tokenizer renders the harmony control tokens (<|message|>/<|call|>) after decode must be
# confirmed against a live capture before this is trusted (see _HARMONY note in parse_tool_calls).
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.([\w.-]+).*?<\|message\|>\s*(\{.*?\})\s*<\|call\|>", re.DOTALL
)

# Markers a streaming consumer watches for to stop emitting text and start
# buffering a tool call. Bare JSON has no marker (handled by a leading-brace check).
# ``<function=`` covers Qwen3-Coder's wrapper-less XML form: without it the raw XML
# streams to the client as visible text even though the call parses at end-of-turn.
# A false positive is safe — unparseable buffered text is flushed back at the end.
TOOL_MARKERS = (_HERMES_OPEN, _PYTHON_TAG, _MISTRAL_TAG, "<function=")


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


def _assign_ids(calls: list[ParsedToolCall]) -> list[ParsedToolCall]:
    """Mint a globally unique id per call. A per-response index (``call_0``…) collides
    across turns: Anthropic-protocol clients (Claude Code) key tool_use/tool_result
    pairs by id over the WHOLE conversation and silently drop repeats, killing every
    tool call after the first turn."""
    for call in calls:
        call.id = f"call_{uuid.uuid4().hex[:24]}"
    return calls


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
    return _assign_ids(calls)


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
    return _assign_ids(calls)


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

    # Nested-XML function form with NO wrapping <tool_call> block. Qwen3-Coder emits
    # ``<function=bash><parameter=command>ls -la</parameter></function>`` directly — sometimes with
    # only a stray closing ``</tool_call>`` and no opener — so _TOOL_CALL_RE never matches and the
    # block at the top is skipped, leaking the call back as text (a real parse_miss caught by the
    # A1 reliability harness). The ``<function=NAME>`` tag is as distinctive as the other markers,
    # so it's safe to trust here without the wrapper.
    if _FUNCTION_RE.search(text):
        calls = _parse_xml_functions(text)
        if calls:
            return calls

    # Harmony (gpt-oss): `to=functions.NAME … <|message|>{json}<|call|>`. Distinctive enough to
    # trust without a wrapper. NOTE: pattern is written to the documented harmony spec but not yet
    # verified against a live gpt-oss capture — confirm and adjust once the model is downloaded.
    if "to=functions." in text and "<|call|>" in text:
        harmony = [
            _coerce_call({"name": name, "arguments": args})
            for name, args in (
                (m.group(1), _first_json(m.group(2)))
                for m in _HARMONY_CALL_RE.finditer(text)
            )
        ]
        harmony = [c for c in harmony if c is not None]
        if harmony:
            return _assign_ids(harmony)

    stripped = text.strip()
    if stripped[:1] in "{[" and known_names:
        return _coerce_all(_iter_json_values(stripped), restrict=known_names)
    return []


def _first_json(s: str):
    """Best-effort: the first JSON value in ``s`` (harmony's <|message|> payload), or {} ."""
    vals = _iter_json_values(s)
    return vals[0] if vals else {}


def earliest_marker(buffer: str, start: int = 0) -> int | None:
    """Index of the first tool-call marker at/after ``start``, or ``None``."""
    best: int | None = None
    for marker in TOOL_MARKERS:
        idx = buffer.find(marker, start)
        if idx != -1 and (best is None or idx < best):
            best = idx
    return best


def _coerce_scalar(value: str, types: list[str]):
    """Coerce a stringified argument to the scalar type its schema declares. Local models over-quote
    non-strings — Qwen3-Coder emitted ``"replace_all": "False"`` (a string) for a boolean param, and
    Claude Code's schema validator then rejected the whole call. Conservative: only touch a value
    that maps CLEANLY, and never when ``string`` is an accepted type (the model may have meant the
    literal text)."""
    if "string" in types:
        return value  # ambiguous — a union with string; leave the literal alone
    low = value.strip().lower()
    if "boolean" in types and low in ("true", "false"):
        return low == "true"
    if "integer" in types:
        try:
            return int(value.strip())
        except ValueError:
            pass
    if "number" in types:
        try:
            return float(value.strip())
        except ValueError:
            pass
    if "null" in types and low in ("null", "none"):
        return None
    return value


def normalize_arguments(arguments: dict, properties: dict) -> dict:
    """Schema-aware coercion of a parsed call's arguments — the normalization middleware shared by
    every consumer (the Anthropic /v1/messages compat route AND the assistant's own agent loop both
    reach it through mlx_service). ``properties`` is the tool's JSON-Schema ``properties`` map; each
    string argument whose param declares a scalar (boolean/integer/number/null) type is coerced to
    it. Unknown params and non-string values pass through untouched, so a well-formed call is never
    altered."""
    if not isinstance(arguments, dict) or not properties:
        return arguments
    out = {}
    for key, val in arguments.items():
        prop = properties.get(key) if isinstance(properties, dict) else None
        if isinstance(val, str) and isinstance(prop, dict):
            declared = prop.get("type")
            types = [declared] if isinstance(declared, str) else list(declared or [])
            out[key] = _coerce_scalar(val, types)
        else:
            out[key] = val
    return out
