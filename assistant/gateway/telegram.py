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
import contextlib
import html as _html
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path

from assistant.gateway.approval import TelegramApprover
from assistant.model_traits import CHATTABLE_KINDS, weak_at_tools

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

# Kinds usable as a chat model — the shared definition in model_traits, not a local copy, so the
# gateway picker can never drift from the GUI picker or the service's load gate (it used to
# mirror mlx_service by hand). Keeps pick_model from auto-selecting a video / embedding / ASR
# model that can't serve a chat turn.
_CHATTABLE_KINDS = CHATTABLE_KINDS

# The weak-at-tools heuristic lives in model_traits (shared with the /models API + GUI picker),
# so the ⚠️ flag is identical everywhere. Aliased to the module-private name the picker uses.
_weak_at_tools = weak_at_tools


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


# Per-tool progress styling (icon, verb), keyed by the tool name on the tool_progress event.
# Unmapped tools fall back to a generic "🛠️ <name>" so a new long tool still renders sensibly
# instead of being mislabelled as the wrong modality (Fusion was showing "Generating video").
_PROGRESS_STYLE = {
    "generate_video": ("🎬", "Generating video"),
    "fusion": ("🔀", "Fusion"),
}


def _progress_bar(fraction: float, label: str = "", *, name: str = "", slots: int = 12) -> str:
    """Render a tool_progress tick as a text bar, e.g. ``🔀 Fusion █████░░░ 42% (panel 1/4: …)``
    or ``🎬 Generating video …``. Long tools (video denoising, the Fusion panel) run for
    minutes; without this the chat looks frozen."""
    fraction = 0.0 if fraction < 0 else 1.0 if fraction > 1 else fraction
    filled = round(fraction * slots)
    bar = "█" * filled + "░" * (slots - filled)
    icon, verb = _PROGRESS_STYLE.get(name, ("🛠️", name or "Working"))
    tail = f" ({label})" if label else ""
    return f"{icon} {verb} {bar} {round(fraction * 100)}%{tail}"


# --- /download progress rendering (N51) ------------------------------------------------------
_DOWNLOAD_POLL_INTERVAL = 3.0  # seconds between snapshot polls (well under Telegram's edit limit)
_DOWNLOAD_WATCH_MAX_TICKS = 2400  # ~2h; a longer download keeps running, just stops being watched


def _human_bytes(n: int) -> str:
    size = float(max(0, n))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _fmt_duration(seconds: int) -> str:
    m, s = divmod(max(0, int(seconds)), 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"


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
        images=None,
        fusion=None,
        model_dirs=None,
        default_workspace=None,
        default_store=None,
        download_manager=None,
    ):
        self._token = token
        self._allowed = set(allowed_users or [])
        self._agent = agent
        self._sessions = sessions
        self._models = model_service
        self._default_model = default_model
        # Live backend default (GUI "Default" → store), preferred over the config snapshot in
        # pick_model so Telegram and the desktop agree on the default model.
        self._default_store = default_store
        self._approval_required = approval_required
        # Optional audio backend (mlx-audio): enables voice-in (STT) and voice-out
        # (TTS). Absent/unavailable -> the gateway stays text-only.
        self._audio = audio
        # Optional video-generation backend (MlxVideoBackend). The model dirs (shared with image
        # discovery below) are scanned for loadable mlx-video checkpoints — together they power
        # the /video picker (N28), which sets the backend's active checkpoint at runtime.
        self._video = video
        self._model_dirs = [Path(d) for d in (model_dirs or [])]
        # Optional image-generation backend (MlxImageBackend). Powers /image (pick the mflux
        # alias OR an on-disk mlx-gen checkpoint discovered from _model_dirs) and /imageset
        # (default size & steps), which set the backend's knobs at runtime.
        self._images = images
        # Optional Fusion engine (panel+judge). Powers /fusion (toggle + pick panel & judge).
        self._fusion = fusion
        # Optional DownloadManager (shared with the GUI/HTTP downloads). Powers /download: submit a
        # HuggingFace repo and watch its progress. Absent -> /download reports it's unavailable.
        self._downloads = download_manager
        self._app = None
        self._pending: dict[str, asyncio.Future] = {}
        # Per-chat model override, set via the /models inline-keyboard picker. Takes
        # precedence over default_model in pick_model, so a Telegram user can switch
        # models without touching config.
        self._selected_model: dict[int, str] = {}
        # Per-chat working directory, set via /cd. Workspace is per-conversation: confirm it
        # before a coding session; all turns in this chat then operate there. Falls back to
        # the server default when unset.
        self._default_workspace = str(default_workspace) if default_workspace else None
        self._workspace: dict[int, str] = {}
        # Per-chat CURRENT conversation id. A chat used to be pinned to one eternal session
        # (`tg:<chat>`), so there was no way to start a fresh topic or go back to an earlier
        # one from Telegram — /new and /sessions manage this pointer. The legacy id stays the
        # default so existing chats keep their history.
        self._session_ids: dict[int, str] = {}
        # Per-chat turn lock: with concurrent_updates(True) two quick messages would otherwise
        # run turns on the same session at once and race its history. See _run_turn.
        self._turn_locks: dict[int, asyncio.Lock] = {}
        # Per-chat in-flight turn task, so /stop can cancel a running turn (B1). The GUI stops
        # via SSE disconnect; Telegram has no such channel, so we track the task and cancel it.
        self._active_turns: dict[int, asyncio.Task] = {}

    # --- lifecycle ---

    async def start(self) -> None:
        if not _PTB_AVAILABLE:
            raise RuntimeError("python-telegram-bot is not installed")
        # concurrent_updates: a coding turn blocks its handler on `await future` while it waits
        # for the user's inline Approve tap. With PTB's default sequential dispatch, that tap's
        # callback update can never be processed (the single worker is busy in the message
        # handler) — so approval always timed out to DENY. Processing updates concurrently lets
        # the approval callback resolve the waiting turn. (Tools needing approval — write/edit/
        # shell — were the first to hit this; media tools don't require approval.)
        app = Application.builder().token(self._token).concurrent_updates(True).build()
        app.add_handler(CommandHandler("start", self._on_start))
        app.add_handler(CommandHandler("whoami", self._on_whoami))
        app.add_handler(CommandHandler("stop", self._on_stop))
        app.add_handler(CommandHandler("models", self._on_models))
        app.add_handler(CommandHandler("new", self._on_new))
        app.add_handler(CommandHandler("sessions", self._on_sessions))
        app.add_handler(CommandHandler("cd", self._on_cd))
        app.add_handler(CommandHandler("download", self._on_download))
        app.add_handler(CommandHandler("video", self._on_video))
        app.add_handler(CommandHandler("videoset", self._on_videoset))
        app.add_handler(CommandHandler("image", self._on_image))
        app.add_handler(CommandHandler("imageset", self._on_imageset))
        app.add_handler(CommandHandler("fusion", self._on_fusion))
        app.add_handler(CallbackQueryHandler(self._on_callback))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message))
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self._on_voice))
        # Inbound images: a photo (or image document), with or without a caption. A photo's
        # caption is NOT filters.TEXT, so this never collides with the text handler above.
        app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_photo))
        await app.initialize()
        # Register slash commands so they appear in Telegram's "/" command menu — without
        # this the user can't discover /models. Non-fatal if the API call fails.
        try:
            await app.bot.set_my_commands(
                [
                    ("start", "Show status and usage"),
                    ("whoami", "Show your Telegram name and id (anyone)"),
                    ("stop", "Stop the turn that's currently running"),
                    ("models", "Pick the chat model"),
                    ("cd", "Set the working directory for this chat"),
                    ("download", "Download a model from HuggingFace"),
                    ("video", "Pick the video-generation model"),
                    ("videoset", "Video defaults: resolution & quality"),
                    ("image", "Pick the image-generation model"),
                    ("imageset", "Image defaults: size & steps"),
                    ("fusion", "Multi-model panel+judge: toggle & pick models"),
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
        # Order of preference: this chat's explicit /models pick, then the live backend
        # default (the GUI's "Default", shared via the store), then an already-loaded model
        # (avoids a load stall), then the first.
        chosen = self._selected_model.get(chat_id) if chat_id is not None else None
        if chosen and any(m.id == chosen for m in models):
            return chosen
        default = self._default_store.value if self._default_store else self._default_model
        if default:
            for m in models:
                if m.id == default:
                    return m.id
        for m in models:
            if m.loaded:
                return m.id
        return models[0].id

    # --- handlers ---

    def _help_html(self, model: str, cwd: str) -> str:
        """The /start status + usage guide. HTML-formatted (Telegram parse_mode=HTML), so any
        interpolated value (model id, path) is escaped. Built from the live command set so the
        guide and the actual handlers stay in step."""
        m, d = _html.escape(model), _html.escape(cwd)
        return (
            "👋 <b>Assistant</b> — a local AI you chat with. It can also generate images &amp; "
            "video, run code, and use tools.\n"
            f"📍 Now: model <b>{m}</b> · dir <code>{d}</code>\n\n"
            "<b>💬 Chat</b>\n"
            "Just send a message. Send a 🎤 voice note to get a spoken reply.\n"
            "/new — start a fresh conversation (the current one is kept)\n"
            "/sessions — list this chat’s saved conversations and switch back to one\n"
            "/models — switch the chat model (⚠️ = weak at tool calls; pick a *-Coder for "
            "coding/agent tasks)\n"
            "/download — fetch a model from HuggingFace, e.g. <code>/download "
            "mlx-community/Qwen2.5-7B-Instruct-4bit</code>; progress streams here\n"
            "/fusion — panel+judge: several models answer and one synthesizes a more accurate "
            "reply. Turn it on, tick the panel + a ⭐ judge, then choose <b>fusion</b> in "
            "/models.\n\n"
            "<b>🖼 Images</b>\n"
            "Just ask, e.g. “a watercolor fox”, “畫一隻貓”. The picture is sent back to you.\n"
            "/image — choose which image model to use\n"
            "/imageset — default size (512/768/1024) &amp; steps (a request like “768x768” "
            "overrides)\n\n"
            "<b>🎬 Video</b>\n"
            "Just ask, e.g. “a drone shot over a forest” (takes a few minutes).\n"
            "/video — choose the video checkpoint\n"
            "/videoset — default resolution &amp; quality\n\n"
            "<b>📂 Files &amp; coding</b>\n"
            "/cd — show this chat’s working directory; <code>/cd ~/proj</code> to change it. The "
            "agent reads, writes and runs commands there and sends back a diff of what changed; "
            "risky actions ask first with Yes/No buttons.\n\n"
            "💡 Images, video and file actions are triggered by <i>asking in plain language</i> — "
            "the commands above only pick which model and settings get used."
        )

    async def _on_start(self, update: "Update", context) -> None:
        uid = update.effective_user.id
        if not self.is_allowed(uid):
            await update.message.reply_text(
                f"Not authorized. Your Telegram user id is {uid} — add it to "
                f"telegram_allowed_users to enable access."
            )
            return
        chat_id = update.effective_chat.id
        model = await self.pick_model(chat_id) or "(none loaded)"
        cwd = self._effective_workspace(chat_id) or "(server default)"
        await update.message.reply_text(
            self._help_html(model, cwd),
            parse_mode="HTML",
            disable_web_page_preview=True,
        )

    async def _on_whoami(self, update: "Update", context) -> None:
        # Deliberately ungated: anyone may call /whoami. The point is to let a not-yet-allowed
        # user read their own id (and copy it) so it can be added to telegram_allowed_users —
        # gating this behind the allowlist would defeat its purpose.
        user = update.effective_user
        name = user.full_name or user.first_name or "there"
        username = f"@{user.username}" if user.username else "None"
        await update.message.reply_text(
            f"Hello <b>{_html.escape(name)}</b>\n"
            f"Your username is <b>{_html.escape(username)}</b>\n"
            f"Your ID is <code>{user.id}</code>",  # <code> = tap-to-copy in Telegram
            parse_mode="HTML",
        )

    async def _on_stop(self, update: "Update", context) -> None:
        # Cancel this chat's in-flight turn (B1). Mirrors the desktop Stop button — useful for a
        # runaway or just-too-slow turn on a large local model. The partial reply is kept.
        if not await self._ensure_allowed(update):
            return
        task = self._active_turns.get(update.effective_chat.id)
        if task is None or task.done():
            await update.message.reply_text("Nothing is running to stop.")
            return
        task.cancel()
        await update.message.reply_text("⏹ Stopping the current turn…")

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
                f"{'● ' if m.id == current else '○ '}{'⚠️ ' if _weak_at_tools(m.id) else ''}{m.id}",
                callback_data=f"model:{i}")]
            for i, m in enumerate(models)
        ]
        note = (
            "Pick the chat model:\n⚠️ = weak at tool calls; for coding use a *-Coder-Instruct."
            if any(_weak_at_tools(m.id) for m in models)
            else "Pick the chat model:"
        )
        await update.message.reply_text(note, reply_markup=InlineKeyboardMarkup(rows))

    def _video_catalog(self):
        # Loadable mlx-video checkpoints across the configured model dirs (cheap filesystem
        # scan). Only converted-MLX Wan/LTX dirs qualify, so the picker never offers a model
        # that would fail at generation time. Stable order → a button index resolves on tap.
        from assistant.models.mlx_discovery import discover_video_checkpoints

        return discover_video_checkpoints(self._model_dirs)

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

    async def _on_download(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._downloads is None:
            await update.message.reply_text("Model download is unavailable on this backend.")
            return
        repo_id = " ".join(context.args or []).strip()
        if not repo_id or any(c.isspace() for c in repo_id):
            # Match the HTTP route's contract: a single 'namespace/name', no spaces/line breaks.
            await update.message.reply_text(
                "Usage: <code>/download namespace/name</code>\n"
                "e.g. <code>/download mlx-community/Qwen2.5-7B-Instruct-4bit</code>",
                parse_mode="HTML",
            )
            return
        # start() is idempotent and resumes a partial. The download runs as its own backend task,
        # so the watch loop below only mirrors progress — ending it never stops the download.
        self._downloads.start(repo_id)
        msg = await update.message.reply_text(self._download_queued_line(repo_id))
        await self._watch_download(context.bot, msg.chat_id, msg.message_id, repo_id)

    async def _watch_download(self, bot, chat_id, message_id, repo_id: str) -> None:
        shown = ""
        terminal = {"done", "error", "cancelled"}

        async def _edit(text: str) -> None:
            nonlocal shown
            if text == shown:  # skip no-op edits (Telegram rejects "message is not modified")
                return
            shown = text
            with contextlib.suppress(Exception):  # transient rate limit / edit failure is non-fatal
                await bot.edit_message_text(text[:4000], chat_id=chat_id, message_id=message_id)

        for _ in range(_DOWNLOAD_WATCH_MAX_TICKS):
            item = next(
                (d for d in self._downloads.snapshot() if d["repo_id"] == repo_id), None
            )
            if item is None:  # removed from the list out from under us
                await _edit(f"⬇️ {repo_id}: no longer tracked.")
                return
            if item["status"] in terminal:
                await _edit(self._download_final_line(repo_id, item))
                return
            # A queued download hasn't started — show "queued", not a "0 B downloaded…" progress
            # line (which looked like it was already transferring). Progress appears once it begins.
            if item["status"] == "queued":
                await _edit(self._download_queued_line(repo_id))
            else:
                await _edit(self._download_progress_line(repo_id, item))
            await asyncio.sleep(_DOWNLOAD_POLL_INTERVAL)
        await _edit(
            f"⬇️ {repo_id}: still downloading in the background — check the app's Downloads."
        )

    @staticmethod
    def _download_queued_line(repo_id: str) -> str:
        # Same text for the initial reply and the queued watch ticks, so _edit dedups (no flicker)
        # until the download actually starts and the progress line takes over.
        return f"⏳ Queued: {repo_id} — waiting for the current download to finish…"

    @staticmethod
    def _download_progress_line(repo_id: str, item: dict) -> str:
        total = item.get("total_bytes") or 0
        done = item.get("downloaded_bytes") or 0
        if total:
            frac = done / total
            filled = round(frac * 12)
            bar = "█" * filled + "░" * (12 - filled)
            line = (
                f"⬇️ {repo_id}\n{bar} {round(frac * 100)}% · "
                f"{_human_bytes(done)} / {_human_bytes(total)}"
            )
        else:  # size unknown (HfApi lookup failed) — show bytes only, no bar
            line = f"⬇️ {repo_id}\n{_human_bytes(done)} downloaded…"
        eta = item.get("eta_seconds")
        if eta:
            line += f" · ETA {_fmt_duration(eta)}"
        return line

    @staticmethod
    def _download_final_line(repo_id: str, item: dict) -> str:
        status = item["status"]
        if status == "done":
            total = item.get("total_bytes") or item.get("downloaded_bytes") or 0
            size = f" ({_human_bytes(total)})" if total else ""
            return f"✅ Downloaded {repo_id}{size}. Pick it with /models."
        if status == "cancelled":
            return f"⏹ Cancelled {repo_id}."
        return f"⚠️ Download failed: {repo_id}\n{_clip_error(item.get('error') or 'unknown error')}"

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

    # --- image generation pickers (/image, /imageset) — mirror the /video pair ---

    def _image_choices(self) -> list[tuple[str, str, str]]:
        """(callback_token, display, value) for the /image picker. value is what set_model
        stores: an mflux alias (schnell/dev) or an mlx-gen checkpoint's absolute path. Built from
        one source so the picker and the tap-handler agree on the token↔value mapping."""
        from assistant.images.mlx_backend import IMAGE_MODELS
        from assistant.models.mlx_discovery import discover_image_checkpoints

        out: list[tuple[str, str, str]] = [(name, name, name) for name in IMAGE_MODELS]
        for m in discover_image_checkpoints(self._model_dirs):
            # token = id (fits Telegram's 64-byte callback_data); display = leaf name; value = path
            out.append((m.id, m.id.split("/")[-1], str(m.path)))
        return out

    def _image_markup(self) -> "InlineKeyboardMarkup":
        current = self._images.model
        rows = [
            [InlineKeyboardButton(
                f"{'● ' if value == current else '○ '}{display}",
                callback_data=f"imodel:{token}")]
            for token, display, value in self._image_choices()
        ]
        return InlineKeyboardMarkup(rows)

    async def _on_image(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._images is None or not self._images.available():
            await update.message.reply_text(
                "Image generation is unavailable (install mflux/mlx-gen on the backend)."
            )
            return
        if not self._image_choices():
            await update.message.reply_text(
                "No image models found. Place an mlx-gen checkpoint (e.g. z-image-turbo) in a "
                "model dir, or set image_model to an mflux alias (schnell/dev)."
            )
            return
        await update.message.reply_text(
            "Pick the image-generation model (on-disk mlx-gen checkpoints generate via the "
            "mlxgen CLI):",
            reply_markup=self._image_markup(),
        )

    # Step presets for /imageset; None ("Default") lets the alias decide (schnell 4 / dev 20).
    _IMAGE_STEP_PRESETS = (("Fast 4", 4), ("Balanced 8", 8), ("Quality 20", 20))

    def _imageset_markup(self) -> "InlineKeyboardMarkup":
        from assistant.images.mlx_backend import IMAGE_SIZES

        cur_size = f"{self._images.size[0]}"  # presets are square → width identifies them
        steps = self._images.steps
        size_row = [
            InlineKeyboardButton(
                f"{'●' if n == cur_size else '○'} {n}", callback_data=f"isize:{n}"
            )
            for n in IMAGE_SIZES
        ]
        step_row = [
            InlineKeyboardButton(
                f"{'●' if v == steps else '○'} {label}", callback_data=f"isteps:{v}"
            )
            for label, v in self._IMAGE_STEP_PRESETS
        ]
        default_row = [
            InlineKeyboardButton(
                f"{'●' if steps is None else '○'} Default steps", callback_data="isteps:0"
            )
        ]
        return InlineKeyboardMarkup([size_row, step_row, default_row])

    async def _on_imageset(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._images is None or not self._images.available():
            await update.message.reply_text(
                "Image generation is unavailable (install mflux on the backend)."
            )
            return
        await update.message.reply_text(
            "Image defaults — tap to change. A request like “512x512” still overrides these.",
            reply_markup=self._imageset_markup(),
        )

    async def _apply_image_choice(self, query) -> None:
        # Point the shared image backend at the chosen model (mflux alias or mlx-gen path).
        # Global, like /video. Resolve the callback token back to its value via _image_choices.
        token = (query.data or "").partition(":")[2]
        if self._images is not None and token:
            value = next((v for t, _, v in self._image_choices() if t == token), token)
            self._images.set_model(value)
        try:
            await query.edit_message_reply_markup(reply_markup=self._image_markup())
        except Exception:
            pass

    async def _apply_imageset_choice(self, query) -> None:
        if self._images is not None:
            kind, _, val = (query.data or "").partition(":")
            if kind == "isize":
                self._images.set_size(val)
            elif kind == "isteps":
                self._images.set_steps(int(val) if val.isdigit() and int(val) > 0 else None)
        try:
            await query.edit_message_reply_markup(reply_markup=self._imageset_markup())
        except Exception:
            pass  # "not modified" (re-tapping the current choice) is non-fatal

    # --- fusion config (/fusion): toggle on/off, multi-select panel, pick judge ---

    async def _fusion_model_ids(self) -> list[str]:
        from assistant.agent.fusion import FUSION_MODEL_ID

        models = await self._models.list_models()
        return [
            m.id for m in models if m.type in _CHATTABLE_KINDS and m.id != FUSION_MODEL_ID
        ]

    def _fusion_markup(self, model_ids: list[str]) -> "InlineKeyboardMarkup":
        cfg = self._fusion.config
        panel, judge, enabled = cfg["panel"], cfg["judge"], cfg["enabled"]
        rows = [[InlineKeyboardButton(
            f"🔀 Fusion: {'ON' if enabled else 'OFF'}", callback_data="fus:toggle")]]
        # One row per model: left toggles panel membership, right sets it as the judge.
        for i, mid in enumerate(model_ids):
            rows.append([
                InlineKeyboardButton(
                    f"{'✅' if mid in panel else '⬜'} {mid}", callback_data=f"fus:panel:{i}"),
                InlineKeyboardButton(
                    f"{'⭐' if mid == judge else '☆'} judge", callback_data=f"fus:judge:{i}"),
            ])
        return InlineKeyboardMarkup(rows)

    async def _on_fusion(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        if self._fusion is None:
            await update.message.reply_text("Fusion is unavailable on this backend.")
            return
        model_ids = await self._fusion_model_ids()
        if not model_ids:
            await update.message.reply_text("No chat models available to build a panel.")
            return
        await update.message.reply_text(
            "Fusion = panel of models + a judge that synthesizes one accurate answer.\n"
            "✅ adds to the panel · ⭐ is the judge. Then pick the “fusion” model with /models.",
            reply_markup=self._fusion_markup(model_ids),
        )

    async def _apply_fusion_choice(self, query) -> None:
        if self._fusion is None:
            return
        _, _, rest = (query.data or "").partition(":")  # strip "fus:"
        kind, _, idx = rest.partition(":")
        model_ids = await self._fusion_model_ids()
        if kind == "toggle":
            self._fusion.configure(enabled=not self._fusion.config["enabled"])
        elif kind == "panel" and idx.isdigit() and int(idx) < len(model_ids):
            mid = model_ids[int(idx)]
            panel = list(self._fusion.config["panel"])
            panel.remove(mid) if mid in panel else panel.append(mid)
            self._fusion.configure(panel=panel)
        elif kind == "judge" and idx.isdigit() and int(idx) < len(model_ids):
            mid = model_ids[int(idx)]
            # Tapping the current judge clears it ("" → None in configure); else set it. (None
            # would mean "no change", so use "" to actually clear.)
            self._fusion.configure(judge="" if mid == self._fusion.config["judge"] else mid)
        try:
            await query.edit_message_reply_markup(reply_markup=self._fusion_markup(model_ids))
        except Exception:
            pass  # "not modified" is non-fatal

    async def _on_message(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        text = update.message.text
        # If this text is a REPLY to a message that carries an image (a photo the user uploaded,
        # or one the bot generated), resolve that image so "make this a sketch" / "edit this" works
        # — a Telegram reply doesn't carry the referenced media into the model's context, so
        # without this the model sees only the text and asks for a path it can't know.
        reply = getattr(update.message, "reply_to_message", None)
        path = await self._download_image(context, reply) if reply is not None else None
        if path is not None:
            text = self._with_image_context(text, path)
        await self._run_turn(update, context, text, voice_reply=False)

    async def _on_photo(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        path = await self._download_image(context, update.message)
        if path is None:
            await update.message.reply_text("Sorry — I couldn't fetch that image.")
            return
        # The caption is the instruction ("make this a sketch"); empty caption -> let the model
        # offer or describe. Either way the image path rides the turn so the model can act on it.
        caption = (update.message.caption or "").strip() or "I've sent you an image."
        await self._run_turn(
            update, context, self._with_image_context(caption, path), voice_reply=False
        )

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

    def _effective_workspace(self, chat_id: int) -> str | None:
        # This chat's /cd choice wins over the server default; None lets the loop use its own
        # configured workspace_dir.
        return self._workspace.get(chat_id) or self._default_workspace

    async def _on_cd(self, update: "Update", context) -> None:
        chat_id = update.effective_chat.id
        arg = " ".join(context.args).strip() if getattr(context, "args", None) else ""
        if not arg:
            cur = self._effective_workspace(chat_id) or "(server default)"
            await update.message.reply_text(
                f"📂 Working directory: {cur}\nUse /cd <path> to change it for this chat."
            )
            return
        path = Path(arg).expanduser()
        if not path.is_dir():
            await update.message.reply_text(f"Not a directory: {path}")
            return
        self._workspace[chat_id] = str(path.resolve())
        await update.message.reply_text(f"📂 Working directory set to {self._workspace[chat_id]}")

    # --- conversations (per-chat sessions) ---

    def _session_id(self, chat_id: int) -> str:
        # Legacy default: a chat that never used /new keeps its original `tg:<chat>` session,
        # so upgrading the backend doesn't orphan an ongoing conversation.
        return self._session_ids.get(chat_id) or f"tg:{chat_id}"

    def _owns_session(self, chat_id: int, session_id: str) -> bool:
        # Exact legacy id, or a `tg:<chat>:<uuid>` conversation started with /new. Matching the
        # trailing ":" matters — without it chat 12 would claim chat 123's conversations.
        return session_id == f"tg:{chat_id}" or session_id.startswith(f"tg:{chat_id}:")

    def _chat_sessions(self, chat_id: int) -> list[dict]:
        """This chat's conversations, most-recently-used first (SessionStore's own order)."""
        return [
            s for s in self._sessions.list_sessions() if self._owns_session(chat_id, s["id"])
        ]

    async def _on_new(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        chat_id = update.effective_chat.id
        # Point at a fresh id without materialising a Session: an empty conversation would
        # otherwise clutter /sessions if the user never sends anything. The first turn creates it.
        self._session_ids[chat_id] = f"tg:{chat_id}:{uuid.uuid4().hex[:8]}"
        await update.message.reply_text(
            "🆕 Started a new conversation. The earlier one is kept — /sessions to go back."
        )

    async def _on_sessions(self, update: "Update", context) -> None:
        if not await self._ensure_allowed(update):
            return
        chat_id = update.effective_chat.id
        sessions = self._chat_sessions(chat_id)
        if not sessions:
            await update.message.reply_text(
                "No saved conversations yet — send a message to start one."
            )
            return
        current = self._session_id(chat_id)
        # Same index-keyed callback_data trick as /models: session ids blow past the 64-byte cap.
        rows = [
            [InlineKeyboardButton(
                f"{'● ' if s['id'] == current else '○ '}{s['title'] or 'Untitled'}"[:60],
                callback_data=f"sess:{i}")]
            for i, s in enumerate(sessions[:10])  # newest 10: a keyboard must stay tappable
        ]
        note = "Pick a conversation to continue:"
        if len(sessions) > 10:
            note += f"\n(showing the 10 most recent of {len(sessions)})"
        await update.message.reply_text(note, reply_markup=InlineKeyboardMarkup(rows))

    async def _apply_session_choice(self, query) -> None:
        _, _, idx = (query.data or "").partition(":")
        chat_id = query.message.chat.id
        sessions = self._chat_sessions(chat_id)
        try:
            chosen = sessions[int(idx)]
        except (ValueError, IndexError):
            await query.edit_message_text("That conversation is no longer available.")
            return
        self._session_ids[chat_id] = chosen["id"]
        with contextlib.suppress(Exception):
            await query.edit_message_text(f"💬 Continuing “{chosen['title'] or 'Untitled'}”")

    def _chat_lock(self, chat_id: int) -> asyncio.Lock:
        lock = self._turn_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._turn_locks[chat_id] = lock
        return lock

    async def _run_turn(
        self, update: "Update", context, text: str, *, voice_reply: bool
    ) -> None:
        chat_id = update.effective_chat.id
        # Serialize turns within a chat: with concurrent_updates(True) a second message could
        # otherwise start a turn on the same session while the first is mid-flight, racing the
        # shared history. Approval callbacks don't go through here, so they stay unblocked.
        async with self._chat_lock(chat_id):
            model = await self.pick_model(chat_id)
            if model is None:
                await update.message.reply_text("No model is available. Load one first.")
                return

            session = self._sessions.get_or_create(self._session_id(chat_id), model=model)
            placeholder = await update.message.reply_text("…")
            editor = _StreamEditor(context.bot, chat_id, placeholder.message_id)
            approver = TelegramApprover(
                self._pending, chat_id, context.bot, self._approval_required
            )
            answer_parts: list[str] = []
            tool_args: dict[str, dict] = {}  # tool_call id -> args, to caption media results
            plan_state: dict = {}  # holds the in-place plan message id for this turn (SA.3)
            # Expose this turn's task so /stop can cancel it (B1). Cleared in the finally below.
            self._active_turns[chat_id] = asyncio.current_task()
            try:
                cwd = self._effective_workspace(chat_id)
                async for ev in self._agent.run(
                    session, text, model, approver=approver, cwd=cwd
                ):
                    if ev["type"] == "assistant_delta":
                        answer_parts.append(ev["content"])
                    await self._handle_event(
                        ev, editor, context.bot, chat_id, tool_args, plan_state
                    )
                await editor.flush(final=True)
            except asyncio.CancelledError:
                # Deliberate /stop: leave the partial reply as-is (shows progress) and end
                # quietly — the /stop handler already acknowledged. Swallow rather than re-raise
                # because this cancellation is our own signal, not shutdown.
                with contextlib.suppress(Exception):
                    await editor.note("⏹ stopped")
                return
            except Exception as exc:
                log.exception("Telegram turn failed")
                await editor.set_error(str(exc))
                return
            finally:
                self._active_turns.pop(chat_id, None)
                # Persist the conversation like the HTTP path does after every turn — without
                # this a Telegram chat lived only in memory and a backend restart silently
                # dropped its whole history. In `finally` so a /stop-cancelled or failed turn
                # still keeps what was said; checkpoint is atomic and best-effort.
                self._sessions.checkpoint(session)
            if voice_reply:
                await self._send_voice_reply(
                    context.bot, chat_id, "".join(answer_parts).strip()
                )

    @staticmethod
    def _image_file_ref(msg) -> tuple[str, str, str] | None:
        """(file_id, unique_id, ext) for an image carried by ``msg``, else None. Telegram sends a
        photo as several renditions; the last is the largest. An uncompressed image sent as a
        document (image/* mime) is handled too."""
        if msg is None:
            return None
        photo = getattr(msg, "photo", None)
        if photo:
            p = photo[-1]
            return (p.file_id, p.file_unique_id, "jpg")
        doc = getattr(msg, "document", None)
        if doc is not None and (getattr(doc, "mime_type", "") or "").startswith("image/"):
            name = getattr(doc, "file_name", "") or ""
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else "png"
            return (doc.file_id, doc.file_unique_id, ext)
        return None

    async def _download_image(self, context, msg) -> str | None:
        """Download the image in ``msg`` to a stable absolute path and return it (None if msg has
        no image). Persisted under the temp dir (not unlinked) so it survives the whole turn AND
        follow-up references ("now make it bigger"); the absolute path lets the agent's edit_image
        / view_image read it regardless of the chat's working directory."""
        ref = self._image_file_ref(msg)
        if ref is None:
            return None
        file_id, uid, ext = ref
        dest = Path(tempfile.gettempdir()) / f"tg_img_{uid}.{ext}"
        try:
            if not dest.exists():
                tg_file = await context.bot.get_file(file_id)
                await tg_file.download_to_drive(str(dest))
            return str(dest)
        except Exception:
            log.exception("telegram image download failed")
            return None

    @staticmethod
    def _with_image_context(text: str, image_path: str) -> str:
        """Ride the resolved image path on the user turn so the model acts on it instead of asking
        for a path it cannot see (Telegram replies/uploads don't reach the model as files)."""
        return (
            f"{text}\n\n[The user attached an image, saved locally at: {image_path}\n"
            "To act on it, call edit_image (to modify it) or view_image (to look at it) with that "
            "exact path — you already have the file, so do not ask the user where it is.]"
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
        if data.startswith("sess:"):  # a /sessions picker tap
            await self._apply_session_choice(query)
            return
        if data.startswith("vchk:"):  # a /video picker tap
            await self._apply_video_choice(query)
            return
        if data.startswith("vres:") or data.startswith("vsteps:"):  # a /videoset tap
            await self._apply_videoset_choice(query)
            return
        if data.startswith("imodel:"):  # an /image picker tap
            await self._apply_image_choice(query)
            return
        if data.startswith("isize:") or data.startswith("isteps:"):  # an /imageset tap
            await self._apply_imageset_choice(query)
            return
        if data.startswith("fus:"):  # a /fusion tap (toggle / panel / judge)
            await self._apply_fusion_choice(query)
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
        self, ev: dict, editor: _StreamEditor, bot, chat_id, tool_args: dict,
        plan_state: dict,
    ) -> None:
        t = ev["type"]
        if t == "assistant_delta":
            editor.add(ev["content"])
            await editor.flush()
        elif t == "tool_call":
            await editor.note(f"⚙️ {ev['name']}…")
            tool_args[ev.get("id")] = ev.get("arguments", {})  # kept to caption the result
        elif t == "tool_progress":
            if ev.get("fraction", 0) < 0:  # heartbeat: indeterminate, show elapsed working time
                await editor.progress(f"🛠️ {ev.get('name', 'working')} … {ev.get('label', '')}")
            else:
                await editor.progress(
                    _progress_bar(ev["fraction"], ev.get("label", ""), name=ev.get("name", ""))
                )
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
            elif name == "text_to_speech":
                await self._send_audio(bot, chat_id, ev["content"])
        elif t == "turn_diff":
            await self._send_diff(bot, chat_id, ev)
        elif t == "plan":
            await self._render_plan(bot, chat_id, ev.get("steps") or [], plan_state)
        elif t == "error":
            await editor.set_error(ev["detail"])

    _PLAN_ICON = {"completed": "✅", "in_progress": "🔄", "pending": "⬜"}

    async def _render_plan(self, bot, chat_id, steps: list[dict], plan_state: dict) -> None:
        # Edit one message in place across the turn's repeated update_plan calls, rather than
        # posting a fresh checklist each time (which would spam the chat). plan_state carries the
        # message id between events of the same turn.
        if not steps:
            return
        lines = ["📋 Plan"] + [
            f"{self._PLAN_ICON.get(s.get('status'), '⬜')} {s.get('title', '')}" for s in steps
        ]
        text = "\n".join(lines)
        mid = plan_state.get("message_id")
        if mid is None:
            try:
                msg = await bot.send_message(chat_id, text)
                plan_state["message_id"] = msg.message_id
            except Exception:
                log.exception("failed to send plan")
        else:
            try:
                await bot.edit_message_text(text, chat_id=chat_id, message_id=mid)
            except Exception:
                pass  # text unchanged or a transient edit failure — not worth surfacing

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

    async def _send_audio(self, bot, chat_id, path: str, caption: str | None = None) -> None:
        # Play a text_to_speech result back as a Telegram voice message (same format mlx-audio
        # produces for _send_voice_reply), mirroring the image/video result routing.
        try:
            with open(path, "rb") as fh:
                await bot.send_voice(
                    chat_id, voice=fh, caption=_clip_caption(caption),
                    read_timeout=120, write_timeout=120, connect_timeout=30,
                )
        except Exception:
            log.exception("failed to send generated audio")

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
