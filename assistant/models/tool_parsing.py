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
* Gemma 4       — ``<|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>``: JSON-ish
                  braces but strings are wrapped in ``<|"|>`` escape tokens, not quotes.
                  Taught by the gemma-4 chat template; mlx-vlm ships a reference parser
                  (``tool_parsers/gemma4.py``) this mirrors.
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

# Plain-<function> hybrid (VibeThinker-3B, N96): an XML-ish wrapper around a Hermes-style
# JSON body — ``<function>\n{"name": …, "arguments": {…}}\n</function>``. Falls in the crack
# between the Hermes branch (needs a <tool_call> wrapper), _FUNCTION_RE (needs ``=NAME`` in
# the tag), and the bare-JSON fallback (the text starts with '<'). Verified against live
# captures: 20/20 sweep runs emitted exactly this shape.
_PLAIN_FUNCTION_OPEN = "<function>"
_PLAIN_FUNCTION_RE = re.compile(r"<function>\s*(.*?)\s*</function>", re.DOTALL)

# Harmony (OpenAI gpt-oss): a tool call rides the "commentary" channel as
#   ...to=functions.NAME <|constrain|>json<|message|>{...json args...}<|call|>
# The recipient carries the tool name; the JSON payload follows <|message|> up to <|call|>.
# UNVERIFIED against real gpt-oss output — the model isn't downloaded yet, and how mlx-lm's
# tokenizer renders the harmony control tokens (<|message|>/<|call|>) after decode must be
# confirmed against a live capture before this is trusted (see _HARMONY note in parse_tool_calls).
# The terminator is ``<|call|>`` OR end-of-text: once <|call|> is registered as a stop
# token (N83) the decoded text ends right BEFORE it, so the payload arrives unterminated.
_HARMONY_CALL_RE = re.compile(
    r"to=functions\.([\w.-]+).*?<\|message\|>\s*(\{.*?\})\s*(?:<\|call\|>|$)", re.DOTALL
)

# Harmony's raw output interleaves channel segments; besides call parsing (below),
# consumers need the TEXT split too: analysis segments are reasoning, final segments
# are the answer, commentary carries tool calls (represented structurally elsewhere).
# Shared by the stream sanitizer (display) and the prompt renderer (history fidelity).
HARMONY_CHANNEL = "<|channel|>"
_HARMONY_SEG_ENDS = ("<|end|>", "<|call|>", "<|return|>", "<|start|>")


def harmony_fields(text: str) -> tuple[str, str]:
    """Split raw harmony output into ``(thinking, content)`` per its channel headers."""
    thinking: list[str] = []
    finals: list[str] = []
    for chunk in text.split(HARMONY_CHANNEL)[1:]:
        header, sep, body = chunk.partition("<|message|>")
        if not sep:
            continue
        cut = min(
            (i for t in _HARMONY_SEG_ENDS if (i := body.find(t)) != -1),
            default=len(body),
        )
        body = body[:cut].strip()
        channel = header.split()[0] if header.split() else ""
        if channel == "analysis" and body:
            thinking.append(body)
        elif channel == "final" and body:
            finals.append(body)
    return "\n\n".join(thinking), "\n\n".join(finals)


# Gemma 4 native syntax, taught by its chat template:
#   <|tool_call>call:NAME{key:<|"|>string<|"|>,n:3,nested:{...},arr:[...]}<tool_call|>
# Strings ride between <|"|> escape tokens instead of JSON quotes; bare literals
# (numbers/booleans/null) parse as JSON. Braces balance manually (no recursive regex)
# so the truncated-generation case degrades to a best-effort parse like the others.
_GEMMA_OPEN = "<|tool_call>"
_GEMMA_ESCAPE = '<|"|>'
_GEMMA_CALL_RE = re.compile(r"call:([\w.-]+)\s*\{")

# Markers a streaming consumer watches for to stop emitting text and start
# buffering a tool call. Bare JSON has no marker (handled by a leading-brace check).
# ``<function=`` covers Qwen3-Coder's wrapper-less XML form: without it the raw XML
# streams to the client as visible text even though the call parses at end-of-turn.
# ``<function>`` (no ``=``) is the plain-JSON hybrid's opener (N96) — same reasoning; for
# emitters that rehearse the block inside <think>, suppression may start mid-reasoning,
# trading a truncated think display for not leaking raw call markup.
# A false positive is safe — unparseable buffered text is flushed back at the end.
TOOL_MARKERS = (
    _HERMES_OPEN,
    _PYTHON_TAG,
    _MISTRAL_TAG,
    "<function=",
    _PLAIN_FUNCTION_OPEN,
    _GEMMA_OPEN,
)


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


def _gemma_matching_brace(text: str, start: int) -> int:
    """Index of the ``}``/``]`` closing the bracket at ``start``, skipping over ``<|"|>``-escaped
    string spans (they may contain braces). Returns ``len(text)`` when unbalanced — a truncated
    generation — so the caller still parses what's there."""
    depth, i = 0, start
    while i < len(text):
        if text.startswith(_GEMMA_ESCAPE, i):
            end = text.find(_GEMMA_ESCAPE, i + len(_GEMMA_ESCAPE))
            if end == -1:
                return len(text)
            i = end + len(_GEMMA_ESCAPE)
            continue
        ch = text[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text)


def _gemma_split_top(text: str) -> list[str]:
    """Split on top-level commas only — not inside nested braces or escaped strings."""
    parts: list[str] = []
    depth, cur, i = 0, 0, 0
    while i < len(text):
        if text.startswith(_GEMMA_ESCAPE, i):
            end = text.find(_GEMMA_ESCAPE, i + len(_GEMMA_ESCAPE))
            i = len(text) if end == -1 else end + len(_GEMMA_ESCAPE)
            continue
        ch = text[i]
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[cur:i])
            cur = i + 1
        i += 1
    if text[cur:].strip():
        parts.append(text[cur:])
    return parts


def _gemma_value(text: str):
    text = text.strip()
    if text.startswith(_GEMMA_ESCAPE):  # escaped string — everything up to the closing escape
        inner = text[len(_GEMMA_ESCAPE) :]
        end = inner.find(_GEMMA_ESCAPE)
        return inner if end == -1 else inner[:end]
    if text.startswith("{"):
        return _gemma_object(text[1 : _gemma_matching_brace(text, 0)])
    if text.startswith("["):
        inner = text[1 : _gemma_matching_brace(text, 0)]
        return [_gemma_value(item) for item in _gemma_split_top(inner) if item.strip()]
    try:  # bare literal: number / boolean / null
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text


def _gemma_object(text: str) -> dict:
    out: dict = {}
    for entry in _gemma_split_top(text):
        colon = entry.find(":")
        if colon == -1:
            continue
        key = entry[:colon].strip()
        if key:
            out[key] = _gemma_value(entry[colon + 1 :])
    return out


def _parse_gemma_calls(text: str) -> list[ParsedToolCall]:
    calls: list[ParsedToolCall] = []
    for m in _GEMMA_CALL_RE.finditer(text):
        brace = m.end() - 1
        close = _gemma_matching_brace(text, brace)
        calls.append(
            ParsedToolCall(
                id="", name=m.group(1), arguments=_gemma_object(text[brace + 1 : close])
            )
        )
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

    # Gemma 4's ``<|tool_call>`` opener is distinctive (the pipe keeps it from ever matching
    # the Hermes ``<tool_call>`` forms above). Brace balancing tolerates a missing closer, so
    # a truncated call still parses; marker seen but nothing parsed falls through to the
    # generic saw-marker discard in the stream consumer.
    if _GEMMA_OPEN in text:
        calls = _parse_gemma_calls(text.split(_GEMMA_OPEN, 1)[1])
        if calls:
            return calls

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

    # Plain ``<function>`` wrapper with a JSON body (VibeThinker-3B, N96) — XML-ish shell,
    # Hermes filling; distinctive enough to trust without known_names (_coerce_call still
    # rejects anything that isn't name+arguments shaped). Think-heavy emitters of this
    # dialect REHEARSE the literal blocks inside <think> — it's plain text to their
    # tokenizer — so only post-reasoning text is scanned: the full buffer would double
    # every call, and an unterminated <think> (ran out of budget mid-reasoning) must
    # yield none, because the model never actually decided. A missing closer on a real
    # block still parses, like the Hermes truncation branch above.
    if _PLAIN_FUNCTION_OPEN in text:
        visible = _strip_think_blocks(text)
        if _PLAIN_FUNCTION_OPEN in visible:
            blocks = _PLAIN_FUNCTION_RE.findall(visible)
            source = "\n".join(blocks) if blocks else visible.split(_PLAIN_FUNCTION_OPEN, 1)[1]
            calls = _coerce_all(_iter_json_values(source), restrict=None)
            if calls:
                return calls

    # Harmony (gpt-oss): `to=functions.NAME … <|message|>{json}` ending in <|call|> or, once
    # <|call|> is a stop token (N83), at end-of-text. Verified against live gpt-oss-120b
    # captures (N82). Distinctive enough to trust without a wrapper.
    if "to=functions." in text and "<|message|>" in text:
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


def _strip_think_blocks(text: str) -> str:
    """Text with ``<think>…</think>`` regions removed. An unterminated ``<think>`` drops
    everything after it: reasoning that never concluded must not contribute tool calls."""
    out: list[str] = []
    rest = text
    while True:
        head, sep, tail = rest.partition("<think>")
        out.append(head)
        if not sep:
            return "".join(out)
        _scratch, sep2, rest = tail.partition("</think>")
        if not sep2:
            return "".join(out)


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
