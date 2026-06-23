from __future__ import annotations

import asyncio
import json
import uuid

from assistant.tools.base import Tool

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    _PTB_AVAILABLE = True
except ImportError:  # python-telegram-bot not installed
    _PTB_AVAILABLE = False


def _fmt_args(arguments: dict, limit: int = 300) -> str:
    try:
        s = json.dumps(arguments, ensure_ascii=False)
    except (TypeError, ValueError):
        s = str(arguments)
    return s if len(s) <= limit else s[:limit] + "…"


class TelegramApprover:
    """Interactive approver for the Telegram gateway.

    For an approval-needing tool it posts inline Approve/Deny buttons and awaits the
    user's tap via a Future that the gateway's callback handler resolves. It times
    out to a safe DENY so a forgotten prompt can never hang the agent loop.

    ``pending`` is the gateway-owned ``{token: Future}`` map shared with the callback
    handler — passing it in (rather than the whole gateway) keeps this unit testable.
    """

    def __init__(
        self,
        pending: dict[str, asyncio.Future],
        chat_id: int,
        bot,
        approval_required: bool,
        timeout: float = 300.0,
    ):
        self._pending = pending
        self._chat = chat_id
        self._bot = bot
        self._required = approval_required
        self._timeout = timeout

    async def approve(self, tool: Tool, arguments: dict) -> bool:
        if not tool.needs_approval:
            return True
        if not self._required:
            return True
        if not _PTB_AVAILABLE:
            return False  # cannot prompt -> fail safe

        token = uuid.uuid4().hex[:8]
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[token] = future
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"ok:{token}"),
                    InlineKeyboardButton("❌ Deny", callback_data=f"no:{token}"),
                ]
            ]
        )
        await self._bot.send_message(
            self._chat,
            f"Approve tool {tool.name}?\n{_fmt_args(arguments)}",
            reply_markup=keyboard,
        )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False
        finally:
            self._pending.pop(token, None)
