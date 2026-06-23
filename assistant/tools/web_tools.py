"""Web tools: search the web and read a page as text.

Dependency-free beyond httpx (already the model/omlx transport): ``web_search`` scrapes
DuckDuckGo's HTML endpoint (no API key), ``fetch_url`` strips a page to readable text.
Both are read-only, so they're not approval-gated — but they DO make outbound network
calls, hence short timeouts and best-effort failure (a clear message, never a hang) so a
flaky network degrades gracefully instead of stalling the agent loop.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import httpx

from .base import ToolContext, ToolResult
from .registry import registry

# A desktop UA — DuckDuckGo's HTML endpoint serves a usable result page to it; an empty
# UA tends to get the JS-only page with no parseable results.
_UA = "Mozilla/5.0 (Macintosh; Apple Silicon) Assistant/0.1"
_TIMEOUT = 12.0
_MAX_FETCH_CHARS = 6000  # cap fetched text so a long page can't blow the token budget


def _clean_ddg(href: str | None) -> str:
    """DuckDuckGo wraps each result URL in a `/l/?uddg=<real-url>` redirect — unwrap it."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return target[0]
    return href


class _DDGParser(HTMLParser):
    """Pull (title, url, snippet) triples out of the DuckDuckGo HTML results page."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._cur: str | None = None  # "title" | "snippet" while inside that anchor
        self._href: str | None = None
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        cls = dict(attrs).get("class") or ""
        if "result__a" in cls:
            self._cur, self._href, self._buf = "title", dict(attrs).get("href"), []
        elif "result__snippet" in cls:
            self._cur, self._buf = "snippet", []

    def handle_data(self, data: str) -> None:
        if self._cur:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._cur:
            return
        text = "".join(self._buf).strip()
        if self._cur == "title":
            self.results.append({"title": text, "url": _clean_ddg(self._href), "snippet": ""})
        elif self._cur == "snippet" and self.results:
            self.results[-1]["snippet"] = text
        self._cur, self._buf = None, []


class _TextExtractor(HTMLParser):
    """Reduce an HTML page to its visible text, dropping script/style/markup."""

    _SKIP = {"script", "style", "noscript", "head", "svg", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


@registry.tool(
    name="web_search",
    description=(
        "Search the web (DuckDuckGo) and return the top results as title + URL + "
        "snippet. Use this to look up current or external information you don't know "
        "(news, weather, prices, docs), then call fetch_url to read a result in full."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "search query"},
            "max_results": {"type": "integer", "description": "1-10, default 5"},
        },
        "required": ["query"],
    },
)
async def web_search(args: dict, ctx: ToolContext) -> ToolResult:
    query = (args.get("query") or "").strip()
    if not query:
        return ToolResult(False, "empty query")
    n = max(1, min(int(args.get("max_results") or 5), 10))
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            resp = await client.post("https://html.duckduckgo.com/html/", data={"q": query})
            resp.raise_for_status()
            html = resp.text
    except Exception as exc:  # network/HTTP — best-effort, surface a clean message
        return ToolResult(False, f"web search failed: {exc}")
    parser = _DDGParser()
    parser.feed(html)
    results = [r for r in parser.results if r["url"]][:n]
    if not results:
        return ToolResult(False, "no results (DuckDuckGo may have rate-limited — retry)")
    lines = [
        f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}".rstrip()
        for i, r in enumerate(results, 1)
    ]
    return ToolResult(True, "\n".join(lines))


@registry.tool(
    name="fetch_url",
    description=(
        "Fetch a web page (or a text/JSON URL) and return its readable text content, "
        "truncated. Use after web_search to read a result in detail."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL"},
            "max_chars": {
                "type": "integer",
                "description": f"cap on returned characters (default {_MAX_FETCH_CHARS})",
            },
        },
        "required": ["url"],
    },
)
async def fetch_url(args: dict, ctx: ToolContext) -> ToolResult:
    url = (args.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return ToolResult(False, "url must start with http:// or https://")
    cap = max(500, min(int(args.get("max_chars") or _MAX_FETCH_CHARS), 20_000))
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            body = resp.text
    except Exception as exc:
        return ToolResult(False, f"fetch failed: {exc}")
    if "html" in ctype.lower():
        extractor = _TextExtractor()
        extractor.feed(body)
        text = "\n".join(extractor.parts).strip()
    else:
        text = body.strip()
    if len(text) > cap:
        text = text[:cap] + f"\n...[truncated, {len(text)} chars total]"
    return ToolResult(True, text or "(empty page)")
