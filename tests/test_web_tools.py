"""Web tools: parser/extraction logic (no network) + registration."""

from __future__ import annotations

import asyncio

from assistant.tools import build_registry
from assistant.tools.base import ToolContext
from assistant.tools.web_tools import _clean_ddg, _DDGParser, _TextExtractor, fetch_url


def test_web_tools_are_registered():
    reg = build_registry()
    assert reg.get("web_search") is not None
    assert reg.get("fetch_url") is not None


def test_clean_ddg_unwraps_redirect():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=x"
    assert _clean_ddg(href) == "https://example.com/a"
    # A direct URL passes through unchanged.
    assert _clean_ddg("https://example.com/b") == "https://example.com/b"


def test_ddg_parser_extracts_title_url_snippet():
    html = """
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fweather.com%2Ftoday">
        Today's weather</a>
      <a class="result__snippet" href="x">Sunny, 25C in Taipei.</a>
    </div>
    """
    p = _DDGParser()
    p.feed(html)
    assert len(p.results) == 1
    assert p.results[0]["title"] == "Today's weather"
    assert p.results[0]["url"] == "https://weather.com/today"
    assert "Sunny" in p.results[0]["snippet"]


def test_text_extractor_drops_script_and_style():
    html = "<html><head><style>.x{}</style></head><body><p>Hello</p>" \
        "<script>var a=1;</script><p>World</p></body></html>"
    ex = _TextExtractor()
    ex.feed(html)
    text = "\n".join(ex.parts)
    assert "Hello" in text and "World" in text
    assert "var a" not in text and ".x{}" not in text


def test_fetch_url_rejects_non_http_without_network():
    res = asyncio.run(fetch_url({"url": "ftp://nope"}, ToolContext(cwd=".")))
    assert res.ok is False
    assert "http" in res.content
