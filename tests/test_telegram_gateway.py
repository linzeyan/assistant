import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from assistant.gateway.approval import TelegramApprover
from assistant.gateway.telegram import TelegramGateway
from assistant.models.types import ModelInfo
from assistant.tools.base import Tool


async def _noop(args, ctx):
    return None


_APPROVAL_TOOL = Tool("write_file", "", {}, _noop, needs_approval=True)
_SAFE_TOOL = Tool("read_file", "", {}, _noop, needs_approval=False)


class _FakeModels:
    def __init__(self, models):
        self._models = models

    async def list_models(self):
        return self._models


class _FakeAudio:
    def __init__(self, available=True, transcript="hello there"):
        self._available = available
        self._transcript = transcript
        self.spoken: list[str] = []

    def available(self):
        return self._available

    async def transcribe(self, path):
        return self._transcript

    async def speak(self, text):
        self.spoken.append(text)
        import pathlib
        import tempfile

        p = pathlib.Path(tempfile.gettempdir()) / "fake_tts.wav"
        p.write_bytes(b"RIFF")  # a real file so _send_voice_reply can open() it
        return p


def _gateway(allowed=None, models=None, default=None, audio=None) -> TelegramGateway:
    return TelegramGateway(
        token="x",
        allowed_users=allowed or [],
        agent=None,
        sessions=None,
        model_service=_FakeModels(models or []),
        default_model=default,
        audio=audio,
    )


# --- allowlist (deny by default) ---


def test_is_allowed_deny_by_default():
    assert _gateway(allowed=[]).is_allowed(123) is False


def test_is_allowed_member():
    assert _gateway(allowed=[123]).is_allowed(123) is True


# --- model selection ---


async def test_pick_model_none_when_no_models():
    assert await _gateway(models=[]).pick_model() is None


async def test_pick_model_prefers_default():
    models = [ModelInfo("a"), ModelInfo("b")]
    assert await _gateway(models=models, default="b").pick_model() == "b"


async def test_pick_model_prefers_loaded():
    models = [ModelInfo("a", loaded=False), ModelInfo("b", loaded=True)]
    assert await _gateway(models=models).pick_model() == "b"


async def test_pick_model_first_fallback():
    models = [ModelInfo("a"), ModelInfo("b")]
    assert await _gateway(models=models).pick_model() == "a"


# --- interactive approver ---


async def test_approver_passes_safe_tool_without_prompting():
    bot = Mock()
    bot.send_message = AsyncMock()
    approver = TelegramApprover({}, 1, bot, approval_required=True)
    assert await approver.approve(_SAFE_TOOL, {}) is True
    bot.send_message.assert_not_awaited()


async def test_approver_auto_approves_when_not_required():
    bot = Mock()
    bot.send_message = AsyncMock()
    approver = TelegramApprover({}, 1, bot, approval_required=False)
    assert await approver.approve(_APPROVAL_TOOL, {}) is True
    bot.send_message.assert_not_awaited()


async def test_approver_resolves_on_user_tap():
    pending: dict = {}
    bot = Mock()
    bot.send_message = AsyncMock()
    approver = TelegramApprover(pending, 1, bot, approval_required=True, timeout=5)

    task = asyncio.create_task(approver.approve(_APPROVAL_TOOL, {"path": "x"}))
    await asyncio.sleep(0.02)  # let approve() register its future
    assert len(pending) == 1
    pending[next(iter(pending))].set_result(True)
    assert await task is True
    bot.send_message.assert_awaited_once()


async def test_approver_denies_on_timeout():
    bot = Mock()
    bot.send_message = AsyncMock()
    approver = TelegramApprover({}, 1, bot, approval_required=True, timeout=0.05)
    assert await approver.approve(_APPROVAL_TOOL, {}) is False


# --- voice in (STT) / voice out (TTS) ---


async def test_transcribe_message_returns_text():
    gw = _gateway(audio=_FakeAudio(transcript="turn on the lights"))
    update = Mock()
    update.message.voice = Mock(file_id="f", file_unique_id="u")
    update.message.audio = None
    context = Mock()
    context.bot.get_file = AsyncMock(return_value=Mock(download_to_drive=AsyncMock()))

    text = await gw._transcribe_message(update, context)
    assert text == "turn on the lights"
    context.bot.get_file.assert_awaited_once_with("f")


async def test_transcribe_message_none_without_media():
    gw = _gateway(audio=_FakeAudio())
    update = Mock()
    update.message.voice = None
    update.message.audio = None
    assert await gw._transcribe_message(update, Mock()) is None


async def test_send_voice_reply_synthesizes_and_sends():
    audio = _FakeAudio()
    gw = _gateway(audio=audio)
    bot = Mock()
    bot.send_voice = AsyncMock()
    await gw._send_voice_reply(bot, 42, "here is your answer")
    assert audio.spoken == ["here is your answer"]
    bot.send_voice.assert_awaited_once()


async def test_send_voice_reply_skips_empty_or_unavailable():
    bot = Mock()
    bot.send_voice = AsyncMock()
    # empty text
    audio = _FakeAudio()
    await _gateway(audio=audio)._send_voice_reply(bot, 42, "")
    # backend unavailable
    await _gateway(audio=_FakeAudio(available=False))._send_voice_reply(bot, 42, "x")
    bot.send_voice.assert_not_awaited()
    assert audio.spoken == []


async def test_on_voice_without_audio_backend_tells_user():
    gw = _gateway(allowed=[7], audio=None)  # audio not configured
    update = Mock()
    update.effective_user.id = 7
    update.message.reply_text = AsyncMock()
    await gw._on_voice(update, Mock())
    update.message.reply_text.assert_awaited_once()
    assert "mlx-audio" in update.message.reply_text.call_args.args[0]
