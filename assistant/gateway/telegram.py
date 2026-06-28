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
import html as _html
import logging
import re
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

# --- Telegram HTML rendering (think collapse + a small markdown subset) ---
# Telegram's HTML parse mode only allows a few tags (<b> <i> <code> <pre> <blockquote>…),
# NOT headings — so ### becomes bold. Everything must be HTML-escaped or the whole message
# is rejected, which is why we render once on the final edit and fall back to plain on error.

_FENCE_RE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*)$")


def _md_inline(s: str) -> str:
    s = _html.escape(s, quote=False)
    s = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", s)  # inline code
    s = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", s)  # bold
    return s


def _md_lines(text: str) -> str:
    out = []
    for ln in text.split("\n"):
        h = _HEADING_RE.match(ln)
        out.append(f"<b>{_md_inline(h.group(1))}</b>" if h else _md_inline(ln))
    return "\n".join(out)


def _md_block(text: str) -> str:
    out, pos = [], 0
    for m in _FENCE_RE.finditer(text):  # ``` fences -> <pre>, escaped
        out.append(_md_lines(text[pos:m.start()]))
        out.append(f"<pre>{_html.escape(m.group(1), quote=False)}</pre>")
        pos = m.end()
    out.append(_md_lines(text[pos:]))
    return "".join(out)


def _render_telegram_html(text: str) -> str:
    """<think> blocks collapse into an expandable blockquote; the rest gets the markdown
    subset Telegram supports. Mirrors the GUI's orphan-</think> handling."""
    out, rest = [], text
    if "</think>" in rest and ("<think>" not in rest or rest.index("</think>") < rest.index("<think>")):
        head, _, rest = rest.partition("</think>")
        out.append(f"<blockquote expandable>{_md_block(head).strip()}</blockquote>")
    while "<think>" in rest:
        before, _, after = rest.partition("<think>")
        out.append(_md_block(before))
        think, sep, rest = after.partition("</think>")
        out.append(f"<blockquote expandable>{_md_block(think).strip()}</blockquote>")
        if not sep:  # unterminated think (a partial stream) — nothing trails it
            rest = ""
    out.append(_md_block(rest))
    return "".join(out)


def _clip_error(detail: str, limit: int = 500) -> str:
    """Cap a user-facing error message. A model-load failure can raise a ValueError that
    lists hundreds of mismatched weight names; surfacing it raw turns the reply into a
    multi-KB key dump. Keep the first ``limit`` chars (full detail stays in the backend log)."""
    detail = detail.strip()
    return detail if len(detail) <= limit else detail[:limit].rstrip() + " […]"


def _same_path(a, b) -> bool:
    """Whether two paths point at the same dir (the /video picker marks the active one)."""
    if a is None or b is None:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def _clip_caption(text: str | None) -> str | None:
    """Telegram caps media captions at 1024 chars; keep the head (the prompt's gist) and
    drop the rest rather than letting send_* reject the whole upload."""
    if not text:
        return None
    text = text.strip()
    return text if len(text) <= 1024 else text[:1021].rstrip() + "…"


def _progress_bar(fraction: float, label: str = "", *, slots: int = 12) -> str:
    """Render a tool_progress tick as a text bar, e.g. ``🎬 Generating video █████░░░ 42%
    (17/40)``. Video generation runs for minutes; without this the chat looks frozen."""
    fraction = 0.0 if fraction < 0 else 1.0 if fraction > 1 else fraction
    filled = round(fraction * slots)
    bar = "█" * filled + "░" * (slots - filled)
    tail = f" ({label})" if label else ""
    return f"🎬 Generating video {bar} {round(fraction * 100)}%{tail}"


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
        await self._edit(text, rich=final)

    async def note(self, line: str) -> None:
        # Transient progress shown only while there's no real content yet, so it
        # never clobbers the streamed answer.
        if self._buf.strip():
            return
        await self._edit(line)

    async def progress(self, line: str) -> None:
        # Like note(), but for long tools that emit many ticks (video denoising): throttled
        # and skip-if-unchanged so we stay under Telegram's edit rate limit. The bar may
        # briefly stand in for any pre-tool preamble; the final flush() restores the answer.
        if line == self._shown:
            return
        if (time.monotonic() - self._last) < self._min:
            return
        self._shown = line
        self._last = time.monotonic()
        await self._edit(line)

    async def set_error(self, detail: str) -> None:
        await self._edit(f"⚠️ {_clip_error(detail)}")

    async def _edit(self, text: str, rich: bool = False) -> None:
        # Only the final message is rich-rendered (think collapsed, markdown/code). Partial
        # streaming stays plain — re-rendering half-open think/fences each tick is fragile,
        # and one malformed tag gets the whole edit rejected.
        if rich:
            try:
                await self._bot.edit_message_text(
                    _render_telegram_html(text)[:4096], chat_id=self._chat,
                    message_id=self._mid, parse_mode="HTML",
                )
                return
            except Exception:
                pass  # bad/over-long HTML -> fall back to the plain edit below
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
        video=None,
        model_dirs=None,
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
        # Optional video-generation backend (MlxVideoBackend) + the model dirs to scan for
        # loadable mlx-video checkpoints — together they power the /video generation-model
        # picker (N28). The picker sets the backend's active checkpoint at runtime.
        self._video = video
        self._video_dirs = [Path(d) for d in (model_dirs or [])]
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
        app.add_handler(CommandHandler("video", self._on_video))
        app.add_handler(CommandHandler("videoset", self._on_videoset))
        app.add_handler(CallbackQueryHandler(self._on_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        await app.initialize()
        # Register slash commands so they appear in Telegram's "/" command menu — without
        # this the user can't discover /models. Non-fatal if the API call fails.
        try:
            await app.bot.set_my_commands(
                [
                    ("start", "Show status and usage"),
                    ("models", "Pick the chat model"),
                    ("video", "Pick the video-generation model"),
                    ("videoset", "Video defaults: resolution & quality"),
                ]
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
            "Ready. Send me a message. Use /models to switch the chat model, "
            "/video to pick the video-generation model."
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

    def _video_catalog(self):
        # Loadable mlx-video checkpoints across the configured model dirs (cheap filesystem
        # scan). Only converted-MLX Wan/LTX dirs qualify, so the picker never offers a model
        # that would fail at generation time. Stable order → a button index resolves on tap.
        from assistant.models.mlx_discovery import discover_video_checkpoints

        return discover_video_checkpoints(self._video_dirs)

    async def _on_video(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._video is None or not self._video.available():
            await update.message.reply_text(
                "Video generation is unavailable (install mlx-video on the backend)."
            )
            return
        catalog = self._video_catalog()
        if not catalog:
            await update.message.reply_text(
                "No video-generation model found. Place a converted-MLX Wan/LTX checkpoint "
                "(e.g. Wan2.2-TI2V-5B-mlx) in a model dir."
            )
            return
        current = self._video.checkpoint
        rows = [
            [InlineKeyboardButton(
                f"{'● ' if _same_path(m.path, current) else '○ '}{m.id}",
                callback_data=f"vchk:{i}")]
            for i, m in enumerate(catalog)
        ]
        await update.message.reply_text(
            "Pick the video-generation model:", reply_markup=InlineKeyboardMarkup(rows)
        )

    # Quality presets for /videoset; None ("Default") lets the checkpoint config decide (≈40).
    _STEP_PRESETS = (("Fast 20", 20), ("Balanced 30", 30), ("Quality 40", 40))

    def _videoset_markup(self) -> "InlineKeyboardMarkup":
        from assistant.models.mlx_video import _RESOLUTIONS

        res, steps = self._video.resolution, self._video.steps
        res_row = [
            InlineKeyboardButton(f"{'●' if n == res else '○'} {n}", callback_data=f"vres:{n}")
            for n in _RESOLUTIONS
        ]
        step_row = [
            InlineKeyboardButton(
                f"{'●' if v == steps else '○'} {label}", callback_data=f"vsteps:{v}"
            )
            for label, v in self._STEP_PRESETS
        ]
        default_row = [
            InlineKeyboardButton(
                f"{'●' if steps is None else '○'} Default steps", callback_data="vsteps:0"
            )
        ]
        return InlineKeyboardMarkup([res_row, step_row, default_row])

    async def _on_videoset(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._video is None or not self._video.available():
            await update.message.reply_text(
                "Video generation is unavailable (install mlx-video on the backend)."
            )
            return
        await update.message.reply_text(
            "Video defaults — tap to change. Lower resolution / fewer steps = faster.\n"
            "(A request like “720p, 5 seconds” still overrides these per clip.)",
            reply_markup=self._videoset_markup(),
        )

    async def _apply_videoset_choice(self, query) -> None:
        # Set the shared backend's default resolution/steps (global, like the /video
        # checkpoint), then refresh just the keyboard so the ● marks track the new state.
        if self._video is not None:
            kind, _, val = (query.data or "").partition(":")
            if kind == "vres":
                self._video.set_resolution(val)
            elif kind == "vsteps":
                self._video.set_steps(int(val) if val.isdigit() and int(val) > 0 else None)
        try:
            await query.edit_message_reply_markup(reply_markup=self._videoset_markup())
        except Exception:
            pass  # "not modified" (re-tapping the current choice) is non-fatal

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
        tool_args: dict[str, dict] = {}  # tool_call id -> arguments, to caption media results
        try:
            async for ev in self._agent.run(session, text, model, approver=approver):
                if ev["type"] == "assistant_delta":
                    answer_parts.append(ev["content"])
                await self._handle_event(ev, editor, context.bot, chat_id, tool_args)
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
        if data.startswith("vchk:"):  # a /video picker tap
            await self._apply_video_choice(query)
            return
        if data.startswith("vres:") or data.startswith("vsteps:"):  # a /videoset tap
            await self._apply_videoset_choice(query)
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

    async def _apply_video_choice(self, query) -> None:
        # Resolve the button index back to a checkpoint (re-scan: same stable order as
        # _on_video) and point the shared video backend at it. Global, not per-chat:
        # generation is heavy and runs one at a time, so a per-chat checkpoint adds no value.
        _, _, idx = (query.data or "").partition(":")
        catalog = self._video_catalog()
        try:
            chosen = catalog[int(idx)]
        except (ValueError, IndexError):
            await query.edit_message_text("That video model is no longer available.")
            return
        if self._video is not None:
            self._video.set_checkpoint(chosen.path)
        try:
            await query.edit_message_text(f"✅ Video model set to {chosen.id}")
        except Exception:
            pass

    async def _handle_event(
        self, ev: dict, editor: _StreamEditor, bot, chat_id, tool_args: dict
    ) -> None:
        t = ev["type"]
        if t == "assistant_delta":
            editor.add(ev["content"])
            await editor.flush()
        elif t == "tool_call":
            await editor.note(f"⚙️ {ev['name']}…")
            tool_args[ev.get("id")] = ev.get("arguments", {})  # kept to caption the result
        elif t == "tool_progress":
            await editor.progress(_progress_bar(ev["fraction"], ev.get("label", "")))
        elif t == "tool_result" and ev["ok"]:
            # Media tools return a saved file path as their content; play each modality
            # back into the chat by mirroring the image path. Non-media results are
            # text-only (folded into the streamed answer), so they aren't routed here.
            name = ev["name"]
            # Caption with the generating prompt so the user can tell which request a clip
            # (which can land minutes later, after other turns) actually belongs to.
            prompt = (tool_args.get(ev.get("id")) or {}).get("prompt")
            if name in ("generate_image", "edit_image"):
                await self._send_photo(bot, chat_id, ev["content"], caption=prompt)
            elif name == "generate_video":
                await self._send_video(bot, chat_id, ev["content"], caption=prompt)
        elif t == "turn_diff":
            await self._send_diff(bot, chat_id, ev)
        elif t == "error":
            await editor.set_error(ev["detail"])

    async def _send_photo(self, bot, chat_id, path: str, caption: str | None = None) -> None:
        try:
            with open(path, "rb") as fh:
                await bot.send_photo(
                    chat_id, photo=fh, caption=_clip_caption(caption),
                    read_timeout=120, write_timeout=120, connect_timeout=30,
                )
        except Exception:
            log.exception("failed to send generated image")

    async def _send_video(self, bot, chat_id, path: str, caption: str | None = None) -> None:
        try:
            with open(path, "rb") as fh:
                # Generous timeouts: a multi-MB upload plus Telegram's server-side video
                # processing routinely overruns the short default read_timeout, which logged
                # the upload as failed even when the clip actually went through (false error).
                await bot.send_video(
                    chat_id, video=fh, caption=_clip_caption(caption),
                    read_timeout=300, write_timeout=300, connect_timeout=30,
                )
        except Exception:
            log.exception("failed to send generated video")

    # Above this, a diff is sent inline as an HTML <pre> block; beyond it (or many files),
    # it goes as a .diff document so it isn't truncated at Telegram's 4096-char message cap.
    _DIFF_INLINE_LIMIT = 3500

    async def _send_diff(self, bot, chat_id, ev: dict) -> None:
        # Return what the agent changed on disk: a short diff inline, a big one as a file.
        summary = ev.get("summary") or "files changed"
        diff = ev.get("diff") or ""
        header = f"✏️ {summary}"
        if not diff:  # e.g. binary-only change — still tell the user what changed
            await self._safe_send_message(bot, chat_id, header)
            return
        if len(diff) <= self._DIFF_INLINE_LIMIT and len(ev.get("files", [])) <= 10:
            html = f"{_html.escape(header)}\n<pre>{_html.escape(diff)}</pre>"
            try:
                await bot.send_message(chat_id, html, parse_mode="HTML")
                return
            except Exception:
                pass  # over-long / bad HTML → fall through to the document path
        tmp = Path(tempfile.gettempdir()) / f"changes_{chat_id}.diff"
        try:
            tmp.write_text(diff, encoding="utf-8")
            with open(tmp, "rb") as fh:
                await bot.send_document(
                    chat_id, document=fh, filename="changes.diff",
                    caption=_clip_caption(header),
                    read_timeout=120, write_timeout=120, connect_timeout=30,
                )
        except Exception:
            log.exception("failed to send turn diff")
        finally:
            tmp.unlink(missing_ok=True)

    async def _safe_send_message(self, bot, chat_id, text: str) -> None:
        try:
            await bot.send_message(chat_id, text)
        except Exception:
            log.exception("failed to send message")
