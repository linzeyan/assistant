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


class _FakeResp:
    def __init__(self, text: str, ctype: str = "text/plain"):
        self.text = text
        self.headers = {"content-type": ctype}

    def raise_for_status(self) -> None:
        pass


class _FakeClient:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        return self._resp


def test_fetch_url_budgets_cjk_by_tokens(monkeypatch):
    # WHY (N94): max_chars protects context TOKENS, and CJK tokenizes ~1 token/char — a plain
    # [:cap] slice admitted ~4x the intended context, and one such page grew a turn by ~20k
    # tokens. The same budget must admit ~4x fewer of the denser characters.
    import assistant.tools.web_tools as wt

    page = "頁" * 4000
    monkeypatch.setattr(wt.httpx, "AsyncClient", lambda **kw: _FakeClient(_FakeResp(page)))
    res = asyncio.run(fetch_url({"url": "https://example.com"}, ToolContext(cwd=".")))
    assert res.ok
    body = res.content.split("\n...[truncated")[0]
    assert len(body) == 1500  # default 6000-char cap -> 1500-token budget -> 1500 CJK chars
    assert "4000 chars total" in res.content


def test_fetch_url_english_cap_unchanged(monkeypatch):
    # WHY: the token-aware cut must not change English behavior — the default cap still
    # returns exactly 6000 ASCII chars (identical to the old char slice).
    import assistant.tools.web_tools as wt

    page = "a" * 10_000
    monkeypatch.setattr(wt.httpx, "AsyncClient", lambda **kw: _FakeClient(_FakeResp(page)))
    res = asyncio.run(fetch_url({"url": "https://example.com"}, ToolContext(cwd=".")))
    body = res.content.split("\n...[truncated")[0]
    assert len(body) == 6000
    assert "10000 chars total" in res.content
