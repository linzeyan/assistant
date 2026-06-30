"""Regression corpus for tool-call parsing (spring5 A1) — death mode 2 locked in.

Each case is a model output in one of the formats local tool-capable models emit, asserting
``parse_tool_calls`` extracts exactly the calls we expect. This is the "sediment failures into a
regression set" half of A1: every real parser miss the live harness (``assistant.eval.measure``)
turns up should be reduced to a row here so it can never silently come back.

The seed cases below are representative captures of the documented formats (Hermes/Qwen, Mistral,
Llama, Qwen nested-XML) plus the shapes that historically broke — a thinking-model ``<think>``
preamble before the call, double-encoded arguments, a truncated block, CJK argument values — and
the death-mode-1 false-positive guards (prose, and a JSON *answer* that must not be read as a call).
They are hand-authored to the formats, not pulled from one specific model run; append real
``model_text`` captures from the harness as they appear.
"""

from __future__ import annotations

import pytest

from assistant.models.tool_parsing import parse_tool_calls

KNOWN = {"web_search", "fetch_url", "list_dir", "read_file", "get_weather"}

# (label, raw model text, known_names, expected [(name, arguments), ...])
CASES = [
    (
        "hermes_single",
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "python latest version"}}\n</tool_call>',
        KNOWN,
        [("web_search", {"query": "python latest version"})],
    ),
    (
        "hermes_two_blocks",
        '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>'
        '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>',
        KNOWN,
        [("list_dir", {"path": "."}), ("read_file", {"path": "README.md"})],
    ),
    (
        "thinking_preamble_then_call",  # reasoning models wrap the call after a <think> block
        "<think>The user wants the weather. I should call get_weather.</think>\n"
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Tokyo"}}</tool_call>',
        KNOWN,
        [("get_weather", {"city": "Tokyo"})],
    ),
    (
        "mistral_tool_calls_array",
        '[TOOL_CALLS][{"name": "web_search", "arguments": {"query": "rust release"}}]',
        KNOWN,
        [("web_search", {"query": "rust release"})],
    ),
    (
        "llama_python_tag_parameters_key",  # Llama uses "parameters", not "arguments"
        '<|python_tag|>{"name": "fetch_url", "parameters": {"url": "https://example.com"}}',
        KNOWN,
        [("fetch_url", {"url": "https://example.com"})],
    ),
    (
        "qwen_nested_xml_no_json",  # Qwen3.x via mlx-vlm: nested XML, carries no JSON
        "<tool_call><function=web_search><parameter=query>台北今天天氣</parameter></function></tool_call>",
        KNOWN,
        [("web_search", {"query": "台北今天天氣"})],
    ),
    (
        # REAL capture (A1 harness, Qwen3-Coder-30B-A3B): the nested-XML form with NO opening
        # <tool_call> — only a stray closing tag. Was a parse_miss until the bare-<function=> branch.
        "qwen_coder_bare_function_stray_close",
        "<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        KNOWN,
        [("bash", {"command": "ls -la"})],
    ),
    (
        "qwen_nested_xml_typed_param",  # XML values that are JSON should recover real types
        "<tool_call><function=read_file><parameter=path>a.txt</parameter>"
        "<parameter=max_lines>50</parameter></function></tool_call>",
        KNOWN,
        [("read_file", {"path": "a.txt", "max_lines": 50})],
    ),
    (
        "double_encoded_arguments",  # some templates emit arguments as a JSON *string*
        '<tool_call>{"name": "web_search", "arguments": "{\\"query\\": \\"who won\\"}"}</tool_call>',
        KNOWN,
        [("web_search", {"query": "who won"})],
    ),
    (
        "truncated_unclosed_block",  # generation cut off before </tool_call>
        '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}',
        KNOWN,
        [("list_dir", {"path": "."})],
    ),
    (
        "cjk_query_value",
        '<tool_call>{"name": "web_search", "arguments": {"query": "東京 天気 今日"}}</tool_call>',
        KNOWN,
        [("web_search", {"query": "東京 天気 今日"})],
    ),
    (
        "bare_json_known_name",  # Llama without the tag — accepted only because the name is known
        '{"name": "web_search", "arguments": {"query": "x"}}',
        KNOWN,
        [("web_search", {"query": "x"})],
    ),
    # --- death-mode-1 false-positive guards: these must parse to NOTHING ---
    (
        "prose_only_no_call",
        "The latest stable Python is 3.13. I didn't need to search for that.",
        KNOWN,
        [],
    ),
    (
        "json_answer_not_a_call",  # a JSON *answer*, no tool name → must not be read as a call
        '{"version": "3.13", "released": "2024-10-07"}',
        KNOWN,
        [],
    ),
    (
        "bare_json_unknown_name_gated",  # right shape, unknown tool, no marker → gated out
        '{"name": "definitely_not_a_tool", "arguments": {}}',
        KNOWN,
        [],
    ),
]


@pytest.mark.parametrize("label, text, known, expected", CASES, ids=[c[0] for c in CASES])
def test_tool_parsing_corpus(label, text, known, expected):
    calls = parse_tool_calls(text, known_names=known)
    assert [(c.name, c.arguments) for c in calls] == expected
