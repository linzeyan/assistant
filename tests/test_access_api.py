"""Who may call the HTTP API (assistant/api/access.py).

The API runs shell commands through the agent. Loopback-only binding doesn't stop a web page:
DNS rebinding lets it reach 127.0.0.1 under its own domain name. So the Host header must name
loopback too, and anything off-machine needs the api token.
"""

from __future__ import annotations

from assistant.config import Settings
from assistant.main import create_app
from fastapi.testclient import TestClient


def _client(tmp_path, client_ip="127.0.0.1", host="127.0.0.1:9981", token=None):
    # Used as a context manager so the lifespan runs and the routes have their state.
    app = create_app(Settings(sessions_dir=tmp_path / "s", models_dir=tmp_path / "m", api_token=token))
    return TestClient(app, base_url=f"http://{host}", client=(client_ip, 50000))


def test_local_client_addressing_loopback_is_allowed(tmp_path):
    for host in ("127.0.0.1:9981", "localhost:9981", "[::1]:9981"):
        with _client(tmp_path, host=host) as c:
            assert c.get("/sessions").status_code == 200


def test_dns_rebinding_page_is_refused(tmp_path):
    # Same loopback connection a rebinding page would make, but under the attacker's name.
    with _client(tmp_path, host="attacker.example:9981") as c:
        assert c.get("/sessions").status_code == 403


def test_remote_client_needs_the_token(tmp_path):
    remote = dict(client_ip="192.168.1.20", host="192.168.1.5:9981")
    with _client(tmp_path, **remote) as c:
        assert c.get("/sessions").status_code == 403  # no token configured
    with _client(tmp_path, token="s3cret", **remote) as c:
        assert c.get("/sessions").status_code == 403
        assert c.get("/sessions", headers={"Authorization": "Bearer wrong"}).status_code == 403
        # Both header shapes real clients send: OpenAI SDKs (Bearer) and Claude Code (x-api-key).
        assert c.get("/sessions", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/sessions", headers={"x-api-key": "s3cret"}).status_code == 200
