from __future__ import annotations

import asyncio
import json
import uuid

from assistant.tools.base import Tool

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, LinkPreviewOptions

    _PTB_AVAILABLE = True
    # Spread into every send/edit that carries model-written text. A link preview makes
    # Telegram's servers fetch the URL, so a model steered into writing
    # https://evil.example/?d=<secret> would leak the secret without anyone tapping it.
    NO_PREVIEW: dict = {"link_preview_options": LinkPreviewOptions(is_disabled=True)}
except ImportError:  # python-telegram-bot not installed
    _PTB_AVAILABLE = False
    NO_PREVIEW = {}

# Arguments up to this long are shown in the prompt itself (Telegram caps a message at 4096);
# longer ones go out as a file. Never cut: a truncated prompt shows a harmless head and hides
# whatever the call does after it.
_INLINE_LIMIT = 3500


def pending_key(chat_id: int, token: str) -> str:
    """Key of a waiting approval in the shared pending map. Scoped by chat so a tap only
    resolves the prompt in the chat where it was asked — the token alone would let an
    allowlisted user in one chat answer a prompt meant for another."""
    return f"{chat_id}:{token}"


def _fmt_args(arguments: dict) -> str:
    try:
        return json.dumps(arguments, ensure_ascii=False, indent=1)
    except (TypeError, ValueError):
        return str(arguments)


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
        key = pending_key(self._chat, token)
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[key] = future
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Approve", callback_data=f"ok:{token}"),
                    InlineKeyboardButton("❌ Deny", callback_data=f"no:{token}"),
                ]
            ]
        )
        args = _fmt_args(arguments)
        if len(args) <= _INLINE_LIMIT:
            await self._bot.send_message(
                self._chat, f"Approve tool {tool.name}?\n{args}",
                reply_markup=keyboard, **NO_PREVIEW,
            )
        else:
            await self._bot.send_document(
                self._chat, document=args.encode(), filename=f"{tool.name}-arguments.json",
                caption=f"Approve tool {tool.name}? Its arguments are {len(args)} characters "
                "— they are in the attached file; read it before deciding.",
                reply_markup=keyboard,
            )
        try:
            return await asyncio.wait_for(future, timeout=self._timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return False
        finally:
            self._pending.pop(key, None)
