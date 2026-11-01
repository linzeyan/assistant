"""A TestClient that reaches the app the way the desktop app does.

The API refuses requests that are neither from loopback with a loopback Host nor carrying the
api token (assistant/api/access.py). TestClient's defaults (Host ``testserver``, client
``testclient``) are neither, so every API test goes through this instead.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def local_client(app, **kwargs) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:9981", client=("127.0.0.1", 50000), **kwargs)
