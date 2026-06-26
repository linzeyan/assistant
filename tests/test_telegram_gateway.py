import asyncio
from unittest.mock import AsyncMock, Mock

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
    models = [ModelInfo("a", type="llm"), ModelInfo("b", type="llm")]
    assert await _gateway(models=models, default="b").pick_model() == "b"


async def test_pick_model_prefers_loaded():
    models = [ModelInfo("a", type="llm", loaded=False), ModelInfo("b", type="llm", loaded=True)]
    assert await _gateway(models=models).pick_model() == "b"


async def test_pick_model_first_fallback():
    models = [ModelInfo("a", type="llm"), ModelInfo("b", type="llm")]
    assert await _gateway(models=models).pick_model() == "a"


async def test_pick_model_skips_non_chat_models():
    # A video model in the catalog must never be auto-selected — loading it as a chat
    # model dies with "No safetensors found". This was the real Telegram failure: a Wan
    # video model fail-classified as "llm" got picked, so even "test" crashed.
    models = [ModelInfo("Wan2.2-S2V-14B", type="video"), ModelInfo("qwen", type="llm")]
    assert await _gateway(models=models).pick_model() == "qwen"


async def test_pick_model_none_when_only_non_chat_models():
    models = [ModelInfo("Wan2.2-S2V-14B", type="video"), ModelInfo("bge", type="embedding")]
    assert await _gateway(models=models).pick_model() is None


# --- per-chat model picker (/models inline keyboard) ---


async def test_pick_model_selected_overrides_default():
    models = [ModelInfo("a", type="llm"), ModelInfo("b", type="llm")]
    gw = _gateway(models=models, default="a")
    gw._selected_model[42] = "b"
    assert await gw.pick_model(42) == "b"  # this chat's pick wins over the default
    assert await gw.pick_model(99) == "a"  # a different chat still gets the default


async def test_pick_model_stale_selection_falls_back():
    # A pick that's since been deleted must not be returned — fall back to normal order.
    gw = _gateway(models=[ModelInfo("a", type="llm")], default="a")
    gw._selected_model[42] = "gone"
    assert await gw.pick_model(42) == "a"


async def test_apply_model_choice_records_pick_by_index():
    # The button index addresses the *chattable* list (video filtered out), so index 1 is
    # "b", not the video model — confirms index<->id stays aligned with the filter.
    models = [ModelInfo("a", type="llm"), ModelInfo("Wan", type="video"), ModelInfo("b", type="llm")]
    gw = _gateway(models=models)
    query = Mock()
    query.data = "model:1"
    query.message.chat.id = 42
    query.edit_message_text = AsyncMock()
    await gw._apply_model_choice(query)
    assert gw._selected_model[42] == "b"
    query.edit_message_text.assert_awaited_once()


async def test_apply_model_choice_handles_stale_index():
    gw = _gateway(models=[ModelInfo("a", type="llm")])
    query = Mock()
    query.data = "model:9"  # out of range (catalog shrank since the menu was shown)
    query.message.chat.id = 42
    query.edit_message_text = AsyncMock()
    await gw._apply_model_choice(query)
    assert 42 not in gw._selected_model  # nothing recorded


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


# --- media result routing (P2: play modalities back into the chat) ---


def _media_result(name, ok=True, content="/x"):
    return {"type": "tool_result", "name": name, "ok": ok, "content": content}


async def test_handle_event_sends_generated_video(tmp_path):
    clip = tmp_path / "out.mp4"
    clip.write_bytes(b"\x00\x00")  # a real file so _send_video can open() it
    bot = Mock()
    bot.send_video = AsyncMock()
    bot.send_photo = AsyncMock()
    await _gateway()._handle_event(
        _media_result("generate_video", content=str(clip)), Mock(), bot, 42
    )
    bot.send_video.assert_awaited_once()
    bot.send_photo.assert_not_awaited()  # routed as video, never as an image


async def test_handle_event_image_routing_unchanged(tmp_path):
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG")
    bot = Mock()
    bot.send_photo = AsyncMock()
    await _gateway()._handle_event(
        _media_result("generate_image", content=str(img)), Mock(), bot, 42
    )
    bot.send_photo.assert_awaited_once()  # the video addition didn't break images


async def test_handle_event_skips_nonmedia_and_failed_results():
    # WHY: only a *successful media* result is a file to play back. A text tool result
    # (folded into the streamed answer) or a failed media call must not try to open a path.
    bot = Mock()
    bot.send_video = AsyncMock()
    bot.send_photo = AsyncMock()
    gw = _gateway()
    await gw._handle_event(_media_result("web_search", content="snippets"), Mock(), bot, 42)
    await gw._handle_event(
        _media_result("generate_video", ok=False, content="/nope"), Mock(), bot, 42
    )
    bot.send_video.assert_not_awaited()
    bot.send_photo.assert_not_awaited()
