"""A TestClient that reaches the app the way the desktop app does.

The API refuses requests that are neither from loopback with a loopback Host and the
per-process client token, nor carrying the api token (assistant/api/access.py). TestClient's
defaults (Host ``testserver``, client ``testclient``, no token) are none of these, so every
API test goes through this instead.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def local_client(app, **kwargs) -> TestClient:
    headers = {"Authorization": f"Bearer {app.state.client_token}", **kwargs.pop("headers", {})}
    return TestClient(
        app, base_url="http://127.0.0.1:9981", client=("127.0.0.1", 50000), headers=headers,
        **kwargs,
    )
