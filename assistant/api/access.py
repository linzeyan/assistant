"""Who may call the HTTP API.

The API can run shell commands through the agent, so "it only listens on 127.0.0.1" is not
enough: any web page can make the browser send requests there, DNS rebinding lets such a page
read the replies, and every other process and account on the Mac can connect too. A request
is let in when it doesn't come from a browser page on another site, and either

- it carries the configured ``api_token`` (remote clients on a LAN bind), or
- it comes from a loopback address AND names a loopback host in ``Host`` (a rebinding page
  connects from loopback too, but its ``Host`` is the attacker's domain) AND carries the
  client token this backend generated at start-up — readable only by the owner's account,
  which is what tells the desktop app apart from any other local process.

The OpenAI/Anthropic compat routes (``/v1/*``) skip the client token: they only run the
model, never a tool, and local coding agents (Claude Code) are pointed at them by URL alone.
"""

from __future__ import annotations

import hmac
import ipaddress
from urllib.parse import urlsplit

from starlette.requests import Request

_LOOPBACK_NAMES = {"127.0.0.1", "localhost", "::1"}


def _host_name(host_header: str) -> str:
    # "127.0.0.1:9981" → "127.0.0.1"; "[::1]:9981" → "::1"; a bare name passes through.
    if host_header.startswith("["):
        return host_header[1:].split("]", 1)[0]
    return host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header


def _is_loopback_ip(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_loopback
    except ValueError:
        return False


def _presented_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    return request.headers.get("x-api-key")


def _matches(presented: str | None, expected: str | None) -> bool:
    return bool(presented and expected) and hmac.compare_digest(
        presented.encode(), expected.encode()
    )


def _from_other_site(request: Request) -> bool:
    """A browser page that isn't served by this backend. Browsers stamp Sec-Fetch-Site on
    every request and Origin on cross-origin ones; native clients send neither."""
    if request.headers.get("sec-fetch-site", "").lower() in ("cross-site", "same-site"):
        return True
    origin = request.headers.get("origin")
    if origin is None:
        return False
    parts = urlsplit(origin)
    return parts.scheme not in ("http", "https") or (parts.hostname or "") not in _LOOPBACK_NAMES


def is_allowed(
    request: Request, api_token: str | None, client_token: str | None = None
) -> bool:
    if _from_other_site(request):
        return False
    presented = _presented_token(request)
    if _matches(presented, api_token):
        return True
    client = request.client.host if request.client else ""
    host = _host_name(request.headers.get("host", "").lower())
    if not (_is_loopback_ip(client) and host in _LOOPBACK_NAMES):
        return False
    return request.url.path.startswith("/v1/") or _matches(presented, client_token)


def is_loopback_bind(backend_host: str) -> bool:
    return backend_host.lower() == "localhost" or _is_loopback_ip(backend_host)
