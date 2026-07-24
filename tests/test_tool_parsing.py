from __future__ import annotations

from assistant.models.tool_parsing import (
    earliest_marker,
    normalize_arguments,
    parse_tool_calls,
)


def test_hermes_single_block():
    text = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.py"}}\n</tool_call>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}
    assert calls[0].id.startswith("call_")


def test_qwen_multiple_blocks_get_distinct_ids():
    text = (
        '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert len({c.id for c in calls}) == 2


def test_ids_are_unique_across_responses():
    # Anthropic-protocol clients (Claude Code) key tool_use/tool_result pairs by id
    # over the whole conversation and silently DROP a tool_use whose id repeats —
    # a per-response counter ("call_0") killed every tool call after the first turn.
    text = '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
    first = parse_tool_calls(text)[0].id
    second = parse_tool_calls(text)[0].id
    assert first != second


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
    assert calls[0].id.startswith("call_")


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


def test_qwen_xml_multiple_calls_get_distinct_ids():
    text = (
        "<tool_call><function=a><parameter=x>1</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=y>2</parameter></function></tool_call>"
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert len({c.id for c in calls}) == 2


def test_bare_xml_function_is_a_stream_marker():
    # Qwen3-Coder emits <function=…> with no <tool_call> wrapper. The parser already
    # handles it; the streaming side must ALSO suppress it, or the raw XML shows up
    # as visible text in the client even though the call executes.
    text = "I'll write the file.\n\n<function=Write>\n<parameter=file_path>a.md</parameter>"
    assert earliest_marker(text) == text.index("<function=")


def test_json_block_still_wins_over_xml_fallback():
    # A JSON-shaped call must parse via the JSON path, not the XML fallback (regression).
    text = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "a.py"}


def test_harmony_gpt_oss_tool_call_is_parsed():
    # gpt-oss rides the harmony "commentary" channel: `to=functions.NAME … <|message|>{json}<|call|>`.
    # (Best-effort until verified against a live gpt-oss capture.)
    text = (
        "analysis I'll write it.<|end|>"
        "commentary to=functions.Write <|constrain|>json"
        '<|message|>{"file_path": "a.md", "content": "hi"}<|call|>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "Write"
    assert calls[0].arguments == {"file_path": "a.md", "content": "hi"}
    assert calls[0].id.startswith("call_")


def test_normalize_coerces_overquoted_scalars_to_schema_types():
    # The bug from the Qwen3-Coder session: `"replace_all": "False"` (a string) made Claude Code's
    # validator reject the Edit call ("expected boolean, got string"). The middleware coerces each
    # over-quoted scalar to the type its schema declares.
    props = {
        "replace_all": {"type": "boolean"},
        "count": {"type": "integer"},
        "ratio": {"type": "number"},
        "parent": {"type": ["string", "null"]},
        "file_path": {"type": "string"},
    }
    args = {
        "replace_all": "False",  # -> False
        "count": "3",  # -> 3
        "ratio": "0.5",  # -> 0.5
        "parent": "None",  # union with string -> left as "None" (ambiguous, don't guess)
        "file_path": "None.md",  # a real string -> untouched
        "unknown": "True",  # not in schema -> untouched
    }
    out = normalize_arguments(args, props)
    assert out["replace_all"] is False
    assert out["count"] == 3
    assert out["ratio"] == 0.5
    assert out["parent"] == "None"  # string union: never guessed away
    assert out["file_path"] == "None.md"
    assert out["unknown"] == "True"


def test_normalize_leaves_correct_calls_and_odd_shapes_untouched():
    # A well-formed call is never altered; missing schema / non-dict args are safe no-ops.
    props = {"flag": {"type": "boolean"}}
    assert normalize_arguments({"flag": True}, props) == {"flag": True}
    assert normalize_arguments({"flag": "maybe"}, props) == {"flag": "maybe"}  # not a clean bool
    assert normalize_arguments({"x": "1"}, {}) == {"x": "1"}  # no schema → untouched


def test_gemma4_call_with_escaped_string_arg():
    # Byte-for-byte the shape the gemma-4 chat template teaches (N80): strings ride
    # between <|"|> escape tokens, not JSON quotes.
    text = '<|tool_call>call:web_search{query:<|"|>latest python<|"|>}<tool_call|>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "latest python"}


def test_gemma4_bare_literals_nested_and_arrays():
    text = (
        "<|tool_call>call:edit{count:3,ratio:0.5,on:true,off:null,"
        'nested:{a:<|"|>x,y<|"|>,b:[1,2]},tags:[<|"|>p<|"|>,<|"|>q<|"|>]}<tool_call|>'
    )
    calls = parse_tool_calls(text)
    assert calls[0].arguments == {
        "count": 3,
        "ratio": 0.5,
        "on": True,
        "off": None,
        "nested": {"a": "x,y", "b": [1, 2]},  # comma inside the escape must not split
        "tags": ["p", "q"],
    }


def test_gemma4_truncated_call_still_parses():
    # EOS-truncated generation: opener present, closer (and closing brace) missing.
    text = '<|tool_call>call:web_search{query:<|"|>latest python<|"|>'
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].arguments == {"query": "latest python"}


def test_gemma4_multiple_calls_get_distinct_ids():
    text = (
        '<|tool_call>call:a{x:<|"|>1<|"|>}<tool_call|>\n'
        '<|tool_call>call:b{y:<|"|>2<|"|>}<tool_call|>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[0].id != calls[1].id


def test_gemma4_opener_is_a_stream_marker():
    # The streaming consumer must buffer from the gemma opener, or raw call syntax
    # leaks to the client as visible text (same failure class as N67's <function=).
    text = 'answer coming <|tool_call>call:a{x:<|"|>1<|"|>}'
    assert earliest_marker(text) == len("answer coming ")


def test_gemma4_does_not_shadow_hermes():
    # The pipe in <|tool_call> must not be confused with Hermes' <tool_call> block.
    text = '<tool_call>{"name": "t", "arguments": {"q": "v"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls[0].name == "t" and calls[0].arguments == {"q": "v"}


def test_harmony_call_without_terminator_parses():
    # With <|call|> registered as a stop token (N83), decoding ends right BEFORE it, so
    # the payload arrives unterminated. It must still parse.
    text = (
        "<|channel|>analysis<|message|>Search first.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.web_search "
        '<|constrain|>json<|message|>{"query": "latest python"}'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "latest python"}


def test_plain_function_json_blocks_parse():
    # VibeThinker-3B (N96): an XML-ish <function> shell around a Hermes JSON body — falls
    # in the crack between the =NAME XML branch and the bare-JSON fallback (text starts
    # with '<'), so 20/20 sweep runs scored "no tool call emitted" while calling tools.
    text = (
        "<function>\n"
        '{"name": "web_search", "arguments": {"query": "latest Python", "max_results": 5}}\n'
        "</function>\n\n<function>\n"
        '{"name": "fetch_url", "arguments": {"url": "https://python.org", "max_chars": 6000}}\n'
        "</function>"
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["web_search", "fetch_url"]
    assert calls[0].arguments["query"] == "latest Python"
    assert len({c.id for c in calls}) == 2


def test_plain_function_rehearsal_inside_think_not_doubled():
    # This dialect's emitters rehearse the literal blocks inside <think> (observed in the
    # live captures) — only the post-reasoning block may produce a call, or every call
    # would double and the rehearsal would run alongside the real one.
    text = (
        '<think>I should call: <function>\n{"name": "web_search", "arguments": '
        '{"query": "x"}}\n</function> yes.</think>\n'
        '<function>\n{"name": "web_search", "arguments": {"query": "x"}}\n</function>'
    )
    calls = parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "web_search"


def test_plain_function_only_inside_unterminated_think_is_no_call():
    # A model that ran out of budget mid-reasoning rehearsed a call but never decided —
    # executing scratch work would act on a decision the model never made.
    text = (
        '<think>maybe <function>\n{"name": "web_search", "arguments": {"query": "x"}}\n'
        "</function> or perhaps"
    )
    assert parse_tool_calls(text) == []


def test_plain_function_truncated_closer_still_parses():
    # Same tolerance as the Hermes opened-but-never-closed branch: a truncated closer
    # must not discard an otherwise complete call.
    calls = parse_tool_calls('<function>\n{"name": "web_search", "arguments": {"query": "x"}}')
    assert [c.name for c in calls] == ["web_search"]


def test_plain_function_garbage_body_is_not_a_call():
    assert parse_tool_calls("<function>not json at all</function>") == []


def test_plain_function_opener_is_a_stream_marker():
    # Without the marker the raw <function> JSON streams to the client as visible text
    # even though the call parses at end-of-turn (same reasoning as ``<function=``).
    assert earliest_marker("prose <function>") == 6


def test_plain_function_does_not_shadow_named_xml_form():
    # ``<function=NAME>`` (Qwen nested-XML) must keep hitting its own branch — the two
    # openers are disjoint literals, and this pins that.
    text = "<function=web_search><parameter=query>taipei</parameter></function>"
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["web_search"]
    assert calls[0].arguments == {"query": "taipei"}
