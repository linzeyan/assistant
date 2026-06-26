"""Telegram gateway: makes the assistant reachable from Telegram.

Runs inside the backend process and reuses the same AgentLoop, sessions, skills,
and memory as the desktop path. Each Telegram chat maps to its own session; replies
stream by editing a single message; tools that need approval prompt with inline
buttons (see TelegramApprover).

Security: the allowlist is deny-by-default — an empty ``allowed_users`` rejects
everyone, and /start tells a user their id so it can be added.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path

from assistant.gateway.approval import TelegramApprover

try:
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
    from telegram.ext import (
        Application,
        CallbackQueryHandler,
        CommandHandler,
        MessageHandler,
        filters,
    )

    _PTB_AVAILABLE = True
except ImportError:
    _PTB_AVAILABLE = False

log = logging.getLogger("assistant.telegram")

# Kinds usable as a chat model (mirror mlx_service._CHATTABLE_KINDS) — keeps pick_model
# from auto-selecting a video / embedding model that can't serve a chat turn.
_CHATTABLE_KINDS = ("llm", "vlm")


class _StreamEditor:
    """Accumulates streamed text into one Telegram message, throttling edits to
    avoid hitting Telegram's per-chat edit rate limits."""

    def __init__(self, bot, chat_id: int, message_id: int, min_interval: float = 1.5):
        self._bot = bot
        self._chat = chat_id
        self._mid = message_id
        self._buf = ""
        self._shown = ""
        self._last = 0.0
        self._min = min_interval

    def add(self, text: str) -> None:
        self._buf += text

    async def flush(self, final: bool = False) -> None:
        if not final and (time.monotonic() - self._last) < self._min:
            return
        text = self._buf.strip() or ("(no response)" if final else "…")
        if text == self._shown:
            return
        self._shown = text
        self._last = time.monotonic()
        await self._edit(text)

    async def note(self, line: str) -> None:
        # Transient progress shown only while there's no real content yet, so it
        # never clobbers the streamed answer.
        if self._buf.strip():
            return
        await self._edit(line)

    async def set_error(self, detail: str) -> None:
        await self._edit(f"⚠️ {detail}")

    async def _edit(self, text: str) -> None:
        try:
            await self._bot.edit_message_text(
                text[:4000], chat_id=self._chat, message_id=self._mid
            )
        except Exception:
            # "message is not modified" / transient rate limits are non-fatal.
            pass


class TelegramGateway:
    def __init__(
        self,
        *,
        token: str,
        allowed_users,
        agent,
        sessions,
        model_service,
        default_model: str | None = None,
        approval_required: bool = True,
        audio=None,
    ):
        self._token = token
        self._allowed = set(allowed_users or [])
        self._agent = agent
        self._sessions = sessions
        self._models = model_service
        self._default_model = default_model
        self._approval_required = approval_required
        # Optional audio backend (mlx-audio): enables voice-in (STT) and voice-out
        # (TTS). Absent/unavailable -> the gateway stays text-only.
        self._audio = audio
        self._app = None
        self._pending: dict[str, asyncio.Future] = {}
        # Per-chat model override, set via the /models inline-keyboard picker. Takes
        # precedence over default_model in pick_model, so a Telegram user can switch
        # models without touching config.
        self._selected_model: dict[int, str] = {}

    # --- lifecycle ---

    async def start(self) -> None:
        if not _PTB_AVAILABLE:
            raise RuntimeError("python-telegram-bot is not installed")
        app = Application.builder().token(self._token).build()
        app.add_handler(CommandHandler("start", self._on_start))
        app.add_handler(CommandHandler("models", self._on_models))
        app.add_handler(CallbackQueryHandler(self._on_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        await app.initialize()
        # Register slash commands so they appear in Telegram's "/" command menu — without
        # this the user can't discover /models. Non-fatal if the API call fails.
        try:
            await app.bot.set_my_commands(
                [("start", "Show status and usage"), ("models", "Pick the chat model")]
            )
        except Exception:
            log.warning("could not register Telegram command menu", exc_info=True)
        await app.start()
        await app.updater.start_polling()
        self._app = app
        log.info("Telegram gateway started")

    async def stop(self) -> None:
        if self._app is None:
            return
        try:
            if self._app.updater:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            log.exception("error stopping Telegram gateway")
        finally:
            self._app = None

    # --- pure-ish helpers (unit tested) ---

    def is_allowed(self, user_id: int) -> bool:
        return user_id in self._allowed  # empty allowlist => deny all

    async def pick_model(self, chat_id: int | None = None) -> str | None:
        # Only text LLMs / VLMs can chat (mirror mlx_service._CHATTABLE_KINDS). A video or
        # embedding model in the catalog must never be auto-picked — loading one as a chat
        # model fails (a Wan video model dies with "No safetensors found").
        models = [m for m in await self._models.list_models() if m.type in _CHATTABLE_KINDS]
        if not models:
            return None
        # Order of preference: this chat's explicit /models pick, then the configured
        # default, then an already-loaded model (avoids a load stall), then the first.
        chosen = self._selected_model.get(chat_id) if chat_id is not None else None
        if chosen and any(m.id == chosen for m in models):
            return chosen
        if self._default_model:
            for m in models:
                if m.id == self._default_model:
                    return m.id
        for m in models:
            if m.loaded:
                return m.id
        return models[0].id

    # --- handlers ---

    async def _on_start(self, update: "Update", context) -> None:
        uid = update.effective_user.id
        if not self.is_allowed(uid):
            await update.message.reply_text(
                f"Not authorized. Your Telegram user id is {uid} — add it to "
                f"telegram_allowed_users to enable access."
            )
            return
        await update.message.reply_text(
            "Ready. Send me a message. Use /models to switch the chat model."
        )

    async def _on_models(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        models = [m for m in await self._models.list_models() if m.type in _CHATTABLE_KINDS]
        if not models:
            await update.message.reply_text("No chat model is available. Load one first.")
            return
        current = await self.pick_model(update.effective_chat.id)
        # callback_data is capped at 64 bytes and model ids blow past that, so key each
        # button by catalog index and resolve it back on tap (the chattable order is stable).
        rows = [
            [InlineKeyboardButton(
                f"{'● ' if m.id == current else '○ '}{m.id}", callback_data=f"model:{i}")]
            for i, m in enumerate(models)
        ]
        await update.message.reply_text(
            "Pick the chat model:", reply_markup=InlineKeyboardMarkup(rows)
        )

    async def _on_message(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        await self._run_turn(update, context, update.message.text, voice_reply=False)

    async def _on_voice(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._audio is None or not self._audio.available():
            await update.message.reply_text(
                "Voice messages need mlx-audio installed on the backend."
            )
            return
        text = await self._transcribe_message(update, context)
        if not text:
            await update.message.reply_text("Sorry — I couldn't transcribe that audio.")
            return
        # Echo the transcription so the user can see what was understood, then answer
        # it and reply by voice as well (voice in -> voice out).
        await update.message.reply_text(f"🎙️ {text}")
        await self._run_turn(update, context, text, voice_reply=True)

    async def _ensure_allowed(self, update: "Update") -> bool:
        uid = update.effective_user.id
        if self.is_allowed(uid):
            return True
        await update.message.reply_text(
            f"Not authorized. Your Telegram user id is {uid}."
        )
        return False

    async def _run_turn(
        self, update: "Update", context, text: str, *, voice_reply: bool
    ) -> None:
        chat_id = update.effective_chat.id
        model = await self.pick_model(chat_id)
        if model is None:
            await update.message.reply_text("No model is available. Load one first.")
            return

        session = self._sessions.get_or_create(f"tg:{chat_id}", model=model)
        placeholder = await update.message.reply_text("…")
        editor = _StreamEditor(context.bot, chat_id, placeholder.message_id)
        approver = TelegramApprover(
            self._pending, chat_id, context.bot, self._approval_required
        )
        answer_parts: list[str] = []
        try:
            async for ev in self._agent.run(session, text, model, approver=approver):
                if ev["type"] == "assistant_delta":
                    answer_parts.append(ev["content"])
                await self._handle_event(ev, editor, context.bot, chat_id)
            await editor.flush(final=True)
        except Exception as exc:
            log.exception("Telegram turn failed")
            await editor.set_error(str(exc))
            return
        if voice_reply:
            await self._send_voice_reply(
                context.bot, chat_id, "".join(answer_parts).strip()
            )

    async def _transcribe_message(self, update: "Update", context) -> str | None:
        media = update.message.voice or update.message.audio
        if media is None:
            return None
        tmp = Path(tempfile.gettempdir()) / f"tg_voice_{media.file_unique_id}.oga"
        try:
            tg_file = await context.bot.get_file(media.file_id)
            await tg_file.download_to_drive(str(tmp))
            text = await self._audio.transcribe(str(tmp))
            return text.strip() or None
        except Exception:
            log.exception("voice transcription failed")
            return None
        finally:
            tmp.unlink(missing_ok=True)

    async def _send_voice_reply(self, bot, chat_id, text: str) -> None:
        if not text or self._audio is None or not self._audio.available():
            return
        try:
            path = await self._audio.speak(text)
            with open(path, "rb") as fh:
                await bot.send_voice(chat_id, voice=fh)
        except Exception:
            log.exception("failed to send voice reply")

    async def _on_callback(self, update: "Update", context) -> None:
        query = update.callback_query
        await query.answer()
        data = query.data or ""
        if data.startswith("model:"):  # a /models picker tap, not an approval
            await self._apply_model_choice(query)
            return
        decision, _, token = data.partition(":")
        future = self._pending.get(token)
        if future and not future.done():
            future.set_result(decision == "ok")
        try:
            await query.edit_message_text(
                "✅ Approved" if decision == "ok" else "❌ Denied"
            )
        except Exception:
            pass

    async def _apply_model_choice(self, query) -> None:
        # Resolve the button's catalog index back to a model id (see _on_models for why we
        # key by index), record it as this chat's pick, and confirm.
        _, _, idx = (query.data or "").partition(":")
        models = [m for m in await self._models.list_models() if m.type in _CHATTABLE_KINDS]
        try:
            chosen = models[int(idx)].id
        except (ValueError, IndexError):
            await query.edit_message_text("That model is no longer available.")
            return
        self._selected_model[query.message.chat.id] = chosen
        try:
            await query.edit_message_text(f"✅ Model set to {chosen}")
        except Exception:
            pass

    async def _handle_event(self, ev: dict, editor: _StreamEditor, bot, chat_id) -> None:
        t = ev["type"]
        if t == "assistant_delta":
            editor.add(ev["content"])
            await editor.flush()
        elif t == "tool_call":
            await editor.note(f"⚙️ {ev['name']}…")
        elif t == "tool_result" and ev["ok"]:
            # Media tools return a saved file path as their content; play each modality
            # back into the chat by mirroring the image path. Non-media results are
            # text-only (folded into the streamed answer), so they aren't routed here.
            name = ev["name"]
            if name in ("generate_image", "edit_image"):
                await self._send_photo(bot, chat_id, ev["content"])
            elif name == "generate_video":
                await self._send_video(bot, chat_id, ev["content"])
        elif t == "error":
            await editor.set_error(ev["detail"])

    async def _send_photo(self, bot, chat_id, path: str) -> None:
        try:
            with open(path, "rb") as fh:
                await bot.send_photo(chat_id, photo=fh)
        except Exception:
            log.exception("failed to send generated image")

    async def _send_video(self, bot, chat_id, path: str) -> None:
        try:
            with open(path, "rb") as fh:
                await bot.send_video(chat_id, video=fh)
        except Exception:
            log.exception("failed to send generated video")
