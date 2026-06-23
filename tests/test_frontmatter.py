from assistant.skills.frontmatter import parse_frontmatter


def test_parses_frontmatter_and_body():
    text = "---\nname: foo\ndescription: bar\ntags: [a, b]\n---\n\n# Title\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "foo"
    assert meta["description"] == "bar"
    assert meta["tags"] == ["a", "b"]
    assert body.startswith("# Title")


def test_no_frontmatter_returns_text_unchanged():
    meta, body = parse_frontmatter("# Just a doc\ncontent")
    assert meta == {}
    assert body == "# Just a doc\ncontent"


def test_body_may_contain_dashes():
    text = "---\nname: x\n---\nbefore\n---\nafter\n"
    meta, body = parse_frontmatter(text)
    assert meta["name"] == "x"
    assert "before" in body and "after" in body


def test_malformed_yaml_degrades_to_empty_meta():
    text = "---\n: : not valid\n---\nbody\n"
    meta, body = parse_frontmatter(text)
    assert meta == {}
    assert "body" in body
