"""Telegram gateway lifecycle (S9).

The gateway is built once at boot, but the Gateways settings let the user change the token or
allowlist at runtime — so we need to (re)start it live, without bouncing the whole backend.
These helpers centralise build / reload / status and the token masking the GUI shows (the full
secret never leaves the backend in an API response). Built deliberately small: one driver
(Telegram) today, but the seam is where Discord/Slack would slot in later.
"""

from __future__ import annotations

import logging

log = logging.getLogger("assistant")


async def build_and_start(settings, app) -> tuple[object | None, str | None]:
    """Construct and start the gateway from current settings. Returns ``(gateway, error)``:
    ``(None, None)`` when no token is configured, ``(None, message)`` when start failed (a bad
    token is common — non-fatal, the backend keeps serving), ``(gateway, None)`` on success."""
    if not settings.telegram_token:
        return None, None
    from assistant.gateway.telegram import TelegramGateway

    gateway = TelegramGateway(
        token=settings.telegram_token,
        allowed_users=settings.telegram_allowed_users,
        agent=app.state.agent,
        sessions=app.state.sessions,
        model_service=app.state.model_service,
        default_model=settings.default_model,
        approval_required=settings.approval_required,
        audio=app.state.audio,
    )
    try:
        await gateway.start()
        return gateway, None
    except Exception as exc:
        log.warning("Telegram gateway failed to start: %s; continuing without it", exc)
        return None, str(exc)


async def reload(app) -> dict:
    """Stop the running gateway (if any) and start a fresh one from current settings. Called by
    PUT /config so a token/allowlist edit applies live."""
    old = getattr(app.state, "telegram", None)
    if old is not None:
        await old.stop()
    gateway, error = await build_and_start(app.state.settings, app)
    app.state.telegram = gateway
    app.state.telegram_error = error
    return status(app)


def mask_token(token: str | None) -> str | None:
    """Show enough to confirm a token is set without leaking it: ``1234…wxyz``."""
    if not token:
        return None
    if len(token) <= 8:
        return "••••"
    return f"{token[:4]}…{token[-4:]}"


def status(app) -> dict:
    s = app.state.settings
    return {
        "telegram_configured": bool(s.telegram_token),
        "telegram_token_masked": mask_token(s.telegram_token),
        "telegram_allowed_users": list(s.telegram_allowed_users),
        "telegram_running": getattr(app.state, "telegram", None) is not None,
        "telegram_error": getattr(app.state, "telegram_error", None),
    }
