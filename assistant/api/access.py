"""Who may call the HTTP API.

The API can run shell commands through the agent, so "it only listens on 127.0.0.1" is not
enough: any web page can make the browser send requests there, and DNS rebinding lets such a
page read the replies. A request is let in when either

- it carries the configured ``api_token`` (remote clients on a LAN bind), or
- it comes from a loopback address AND names a loopback host in ``Host``. A rebinding page
  connects from loopback too, but its ``Host`` is the attacker's domain.
"""

from __future__ import annotations

import hmac
import ipaddress

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


def is_allowed(request: Request, api_token: str | None) -> bool:
    if api_token:
        presented = _presented_token(request)
        if presented and hmac.compare_digest(presented.encode(), api_token.encode()):
            return True
    client = request.client.host if request.client else ""
    host = _host_name(request.headers.get("host", "").lower())
    return _is_loopback_ip(client) and host in _LOOPBACK_NAMES


def is_loopback_bind(backend_host: str) -> bool:
    return backend_host.lower() == "localhost" or _is_loopback_ip(backend_host)
