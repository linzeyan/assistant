"""Gateway lifecycle (S9): masked-token status + live reload (stop old, start new). The
masking matters because the full token must never leave the backend in an API response."""

from __future__ import annotations

from types import SimpleNamespace

import assistant.gateway.lifecycle as lifecycle
from assistant.config import Settings


def _app(settings: Settings, telegram=None):
    return SimpleNamespace(state=SimpleNamespace(settings=settings, telegram=telegram))


def test_mask_token_hides_the_secret():
    assert lifecycle.mask_token(None) is None
    assert lifecycle.mask_token("") is None
    assert lifecycle.mask_token("shorty") == "••••"  # too short to safely show any
    assert lifecycle.mask_token("123456789:ABCdefXYZ") == "1234…fXYZ"


def test_status_no_token():
    st = lifecycle.status(_app(Settings(telegram_token=None)))
    assert st == {
        "telegram_configured": False,
        "telegram_token_masked": None,
        "telegram_allowed_users": [],
        "telegram_running": False,
        "telegram_error": None,
    }


def test_status_masks_configured_token():
    app = _app(Settings(telegram_token="123456789:ABCdefXYZ", telegram_allowed_users=[7]))
    st = lifecycle.status(app)
    assert st["telegram_configured"] and st["telegram_token_masked"] == "1234…fXYZ"
    assert st["telegram_allowed_users"] == [7]
    assert st["telegram_running"] is False  # nothing started in this bare app


async def test_reload_stops_old_and_starts_new(monkeypatch):
    stopped = {"v": False}

    class FakeOld:
        async def stop(self):
            stopped["v"] = True

    new_gateway = object()

    async def fake_build(settings, app):
        return new_gateway, None

    monkeypatch.setattr(lifecycle, "build_and_start", fake_build)
    app = _app(Settings(telegram_token="t"), telegram=FakeOld())
    st = await lifecycle.reload(app)
    assert stopped["v"]  # the old gateway was stopped before replacing it
    assert app.state.telegram is new_gateway
    assert st["telegram_running"]


async def test_reload_clears_gateway_when_token_removed(monkeypatch):
    class FakeOld:
        async def stop(self):
            pass

    async def fake_build(settings, app):
        return None, None  # no token -> nothing to start

    monkeypatch.setattr(lifecycle, "build_and_start", fake_build)
    app = _app(Settings(telegram_token=None), telegram=FakeOld())
    st = await lifecycle.reload(app)
    assert app.state.telegram is None and not st["telegram_running"]
