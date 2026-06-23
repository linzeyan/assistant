from __future__ import annotations

from assistant.models.tool_parsing import earliest_marker, parse_tool_calls


def test_hermes_single_block():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert calls[0].id == "call_0"


def test_qwen_multiple_blocks_get_sequential_ids():
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [(c.id, c.name) for c in calls] == [("call_0", "a"), ("call_1", "b")]


def test_mistral_tool_calls_array():
    text = '[TOOL_CALLS][{"name": "search", "arguments": {"q": "x"}}]'
    calls = parse_tool_calls(text)
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "x"}


def test_llama_python_tag_with_parameters_alias():
    text = '<|python_tag|>{"name": "bash", "parameters": {"cmd": "ls"}}'
    calls = parse_tool_calls(text)
    assert calls[0].name == "bash"
    assert calls[0].arguments == {"cmd": "ls"}


def test_arguments_double_encoded_as_string():
    text = '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0].arguments == {"a": 1}


def test_prose_is_never_a_tool_call():
    assert parse_tool_calls("Here is the answer: 42.") == []


def test_malformed_block_yields_nothing():
    assert parse_tool_calls("<tool_call>not json at all</tool_call>") == []


def test_unterminated_block_still_parses():
    # A truncated generation may open <tool_call> without closing it.
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "a"}}'
    calls = parse_tool_calls(text)
    assert calls and calls[0].name == "read_file"


def test_bare_json_requires_a_known_tool_name():
    text = '{"name": "write_file", "arguments": {"path": "x", "content": "y"}}'
    assert parse_tool_calls(text, known_names={"write_file"})[0].name == "write_file"
    # Unknown name, or no allowlist at all -> treat as a plain JSON answer, not a call.
    assert parse_tool_calls(text, known_names={"other"}) == []
    assert parse_tool_calls(text) == []


def test_prose_then_marker_position():
    text = "Let me look. <tool_call>{}"
    assert earliest_marker(text) == text.index("<tool_call>")
    assert earliest_marker("no markers here") is None


def test_qwen_xml_function_call_is_parsed():
    # Qwen3.x (e.g. via mlx-vlm) emits nested XML with no JSON inside <tool_call>.
    text = (
        "<tool_call>\n<function=web_search>\n"
        "<parameter=query>\n台北今天天氣\n</parameter>\n</function>\n</tool_call>"
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "台北今天天氣"}
    assert calls[0].id == "call_0"


def test_qwen_xml_multiple_params_and_type_recovery():
    text = (
        "<tool_call><function=read_file>"
        "<parameter=path>a.py</parameter>"
        "<parameter=start>10</parameter>"
        "</function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert calls[0].name == "read_file"
    # A plain string stays a string; a JSON-looking scalar recovers its real type.
    assert calls[0].arguments == {"path": "a.py", "start": 10}


def test_qwen_xml_multiple_calls_get_sequential_ids():
    text = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [(c.id, c.name) for c in calls] == [("call_0", "a"), ("call_1", "b")]


def test_json_block_still_wins_over_xml_fallback():
    # A JSON-shaped call must parse via the JSON path, not the XML fallback (regression).
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
