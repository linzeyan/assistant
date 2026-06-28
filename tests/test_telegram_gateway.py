import asyncio
from unittest.mock import AsyncMock, Mock

from assistant.gateway.approval import TelegramApprover
from assistant.gateway.telegram import (
    TelegramGateway,
    _clip_caption,
    _clip_error,
    _progress_bar,
    _render_telegram_html,
    _weak_at_tools,
)
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


class _FakeVideo:
    def __init__(self, available=True, checkpoint=None):
        self._available = available
        self._checkpoint = checkpoint
        self.set_to = "UNSET"
        self.resolution = "360p"
        self.steps = None

    def available(self):
        return self._available

    @property
    def checkpoint(self):
        return self._checkpoint

    def set_checkpoint(self, p):
        self._checkpoint = p
        self.set_to = p

    def set_resolution(self, name):
        self.resolution = name

    def set_steps(self, steps):
        self.steps = steps


def _gateway(
    allowed=None, models=None, default=None, audio=None, video=None, model_dirs=None
) -> TelegramGateway:
    return TelegramGateway(
        token="x",
        allowed_users=allowed or [],
        agent=None,
        sessions=None,
        model_service=_FakeModels(models or []),
        default_model=default,
        audio=audio,
        video=video,
        model_dirs=model_dirs,
    )


def _make_mlx_video_ckpt(d):
    import json

    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    for f in ("vae.safetensors", "t5_encoder.safetensors", "model.safetensors"):
        (d / f).write_bytes(b"\x00")
    return d


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


# --- Telegram HTML rendering (think collapse + markdown subset, N30) ---


def test_render_collapses_think_into_blockquote():
    out = _render_telegram_html("<think>reasoning here</think>the answer")
    assert "<blockquote expandable>reasoning here</blockquote>" in out
    assert out.rstrip().endswith("the answer")


def test_render_handles_orphan_think():
    # Qwen3.x templates inject the opener, so the stream often starts mid-think.
    out = _render_telegram_html("just reasoning</think>final answer")
    assert out.startswith("<blockquote expandable>just reasoning</blockquote>")
    assert "final answer" in out


def test_render_code_fence_becomes_pre_and_escapes():
    out = _render_telegram_html("```go\nif a < b && c > d {}\n```")
    assert "<pre>" in out and "</pre>" in out
    assert "&lt;" in out and "&gt;" in out and "&amp;" in out  # escaped inside <pre>


def test_render_headings_and_bold():
    out = _render_telegram_html("### Title\nsome **bold** text")
    assert "<b>Title</b>" in out and "<b>bold</b>" in out


def test_render_escapes_stray_angle_brackets():
    # A bare < in prose must be escaped or Telegram rejects the whole message.
    out = _render_telegram_html("compare a < b")
    assert "&lt;" in out and "<blockquote" not in out


def test_video_catalog_lists_only_loadable_checkpoints(tmp_path):
    # The /video picker offers a converted-MLX checkpoint but NOT a raw HF Wan dir (model_type
    # ti2v yet no vae/t5_encoder) — that would classify as video but fail at generation time.
    import json

    _make_mlx_video_ckpt(tmp_path / "Wan2.2-TI2V-5B-mlx")
    raw = tmp_path / "Wan2.2-TI2V-5B"
    raw.mkdir()
    (raw / "config.json").write_text(json.dumps({"model_type": "ti2v"}))
    (raw / "diffusion_pytorch_model-00001.safetensors").write_bytes(b"\x00")
    gw = _gateway(video=_FakeVideo(), model_dirs=[tmp_path])
    assert [m.id for m in gw._video_catalog()] == ["Wan2.2-TI2V-5B-mlx"]


async def test_apply_video_choice_points_backend_at_checkpoint(tmp_path):
    ckpt = _make_mlx_video_ckpt(tmp_path / "Wan-mlx")
    video = _FakeVideo()
    gw = _gateway(video=video, model_dirs=[tmp_path])
    query = Mock()
    query.data = "vchk:0"
    query.edit_message_text = AsyncMock()
    await gw._apply_video_choice(query)
    assert video.set_to == ckpt  # backend now points at the chosen checkpoint
    query.edit_message_text.assert_awaited()


async def test_apply_video_choice_handles_stale_index(tmp_path):
    video = _FakeVideo()
    gw = _gateway(video=video, model_dirs=[tmp_path])  # empty dir → no checkpoints
    query = Mock()
    query.data = "vchk:0"
    query.edit_message_text = AsyncMock()
    await gw._apply_video_choice(query)
    assert video.set_to == "UNSET"  # nothing applied
    query.edit_message_text.assert_awaited_with("That video model is no longer available.")


async def test_videoset_taps_set_backend_defaults(tmp_path):
    # WHY: /videoset buttons set the shared backend's default resolution/steps (a request
    # still overrides per clip); the menu refreshes so ● tracks the new state.
    video = _FakeVideo()
    gw = _gateway(video=video, model_dirs=[tmp_path])
    query = Mock()
    query.edit_message_reply_markup = AsyncMock()

    query.data = "vres:720p"
    await gw._apply_videoset_choice(query)
    assert video.resolution == "720p"

    query.data = "vsteps:20"
    await gw._apply_videoset_choice(query)
    assert video.steps == 20

    query.data = "vsteps:0"  # "Default steps" → back to config default (None)
    await gw._apply_videoset_choice(query)
    assert video.steps is None
    query.edit_message_reply_markup.assert_awaited()


def test_videoset_markup_marks_current_resolution(tmp_path):
    video = _FakeVideo()
    video.resolution = "480p"
    gw = _gateway(video=video, model_dirs=[tmp_path])
    labels = [b.text for row in gw._videoset_markup().inline_keyboard for b in row]
    assert "● 480p" in labels and "○ 360p" in labels


async def test_send_diff_small_goes_inline_as_pre():
    bot = Mock()
    bot.send_message = AsyncMock()
    bot.send_document = AsyncMock()
    await _gateway()._send_diff(
        bot, 42, {"summary": "1 file changed (+1/-0)", "files": [{"path": "a"}], "diff": "+hi\n"}
    )
    bot.send_message.assert_awaited_once()
    args, kwargs = bot.send_message.await_args
    assert kwargs.get("parse_mode") == "HTML" and "<pre>" in args[1]
    bot.send_document.assert_not_awaited()


async def test_send_diff_large_goes_as_document():
    bot = Mock()
    bot.send_message = AsyncMock()
    bot.send_document = AsyncMock()
    big = "+x\n" * 2000  # well over the inline limit → must be sent as a .diff file
    await _gateway()._send_diff(bot, 42, {"summary": "big", "files": [{"path": "a"}], "diff": big})
    bot.send_document.assert_awaited_once()
    assert bot.send_document.await_args.kwargs["filename"] == "changes.diff"
    bot.send_message.assert_not_awaited()


async def test_on_cd_sets_per_chat_workspace(tmp_path):
    gw = _gateway()
    update = Mock()
    update.effective_chat.id = 42
    update.message.reply_text = AsyncMock()
    context = Mock()
    context.args = [str(tmp_path)]
    await gw._on_cd(update, context)
    assert gw._workspace[42] == str(tmp_path.resolve())


async def test_on_cd_rejects_non_directory():
    gw = _gateway()
    update = Mock()
    update.effective_chat.id = 42
    update.message.reply_text = AsyncMock()
    context = Mock()
    context.args = ["/no/such/dir/zzz"]
    await gw._on_cd(update, context)
    assert 42 not in gw._workspace  # nothing recorded
    assert "Not a directory" in update.message.reply_text.call_args.args[0]


def test_effective_workspace_prefers_per_chat_over_default():
    gw = _gateway()
    gw._default_workspace = "/srv/default"
    assert gw._effective_workspace(42) == "/srv/default"  # falls back to server default
    gw._workspace[42] = "/home/proj"
    assert gw._effective_workspace(42) == "/home/proj"  # this chat's /cd wins


def test_weak_at_tools_flags_thinking_models_but_not_coder():
    # The two models that fabricated "git diff" live must be flagged...
    assert _weak_at_tools("mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit")
    assert _weak_at_tools("mlx-community/Qwen3-30B-A3B-8bit")
    # ...but the tool-caller to prefer must NOT be (it's also -30B-A3B).
    assert not _weak_at_tools("mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit")
    assert not _weak_at_tools("mlx-community/Qwen2.5-7B-Instruct")


async def test_models_picker_marks_weak_models(tmp_path):
    models = [ModelInfo("Qwen3-Coder-30B-A3B-Instruct", type="llm"),
              ModelInfo("DeepSeek-R1-Distill-Qwen-32B", type="llm")]
    gw = _gateway(models=models)
    update = Mock()
    update.effective_chat.id = 42
    update.effective_user.id = 7
    update.message.reply_text = AsyncMock()
    gw._allowed = {7}
    await gw._on_models(update, Mock())
    kwargs = update.message.reply_text.await_args.kwargs
    labels = [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert any("⚠️" in lbl and "DeepSeek-R1" in lbl for lbl in labels)
    assert not any("⚠️" in lbl and "Coder" in lbl for lbl in labels)


def test_clip_error_caps_runaway_dump():
    # A model-load ValueError can list hundreds of weight names (Lance-3B-Video dumped
    # 1606); the reply must stay short, not become a multi-KB key dump.
    short = "model failed to load: bad arch"
    assert _clip_error(short) == short  # normal errors pass through untouched
    runaway = "Received 1606 parameters not in model: " + ", ".join(
        f"blocks.{i}.attn.qkv.weight" for i in range(400)
    )
    clipped = _clip_error(runaway)
    assert len(clipped) <= 510 and clipped.endswith("[…]")
    assert clipped.startswith("Received 1606 parameters")


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


async def test_on_callback_approve_resolves_pending_future():
    # WHY: the live deadlock (approval taps did nothing) hid that _on_callback's ok:/no:
    # routing was never unit-tested. Pin it: an "ok:<token>" tap resolves that token's
    # waiting future to True so the agent loop proceeds.
    gw = _gateway()
    fut = asyncio.get_event_loop().create_future()
    gw._pending["abc123"] = fut
    query = Mock()
    query.data = "ok:abc123"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = Mock()
    update.callback_query = query
    await gw._on_callback(update, Mock())
    assert fut.result() is True


async def test_on_callback_deny_resolves_future_false():
    gw = _gateway()
    fut = asyncio.get_event_loop().create_future()
    gw._pending["abc123"] = fut
    query = Mock()
    query.data = "no:abc123"
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    update = Mock()
    update.callback_query = query
    await gw._on_callback(update, Mock())
    assert fut.result() is False


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
        _media_result("generate_video", content=str(clip)), Mock(), bot, 42, {}
    )
    bot.send_video.assert_awaited_once()
    bot.send_photo.assert_not_awaited()  # routed as video, never as an image


async def test_handle_event_sends_text_to_speech_as_voice(tmp_path):
    clip = tmp_path / "out.wav"
    clip.write_bytes(b"RIFF")  # a real file so _send_audio can open() it
    bot = Mock()
    bot.send_voice = AsyncMock()
    bot.send_photo = AsyncMock()
    await _gateway()._handle_event(
        _media_result("text_to_speech", content=str(clip)), Mock(), bot, 42, {}
    )
    bot.send_voice.assert_awaited_once()  # routed as a voice message
    bot.send_photo.assert_not_awaited()


async def test_handle_event_image_routing_unchanged(tmp_path):
    img = tmp_path / "out.png"
    img.write_bytes(b"\x89PNG")
    bot = Mock()
    bot.send_photo = AsyncMock()
    await _gateway()._handle_event(
        _media_result("generate_image", content=str(img)), Mock(), bot, 42, {}
    )
    bot.send_photo.assert_awaited_once()  # the video addition didn't break images


async def test_handle_event_skips_nonmedia_and_failed_results():
    # WHY: only a *successful media* result is a file to play back. A text tool result
    # (folded into the streamed answer) or a failed media call must not try to open a path.
    bot = Mock()
    bot.send_video = AsyncMock()
    bot.send_photo = AsyncMock()
    gw = _gateway()
    await gw._handle_event(_media_result("web_search", content="snippets"), Mock(), bot, 42, {})
    await gw._handle_event(
        _media_result("generate_video", ok=False, content="/nope"), Mock(), bot, 42, {}
    )
    bot.send_video.assert_not_awaited()
    bot.send_photo.assert_not_awaited()


async def test_handle_event_captions_video_with_its_prompt(tmp_path):
    # WHY: a clip can arrive minutes later (after other turns), so the user can't tell which
    # request it answers. The tool_call's prompt is remembered and sent as the video caption.
    clip = tmp_path / "out.mp4"
    clip.write_bytes(b"\x00\x00")
    bot = Mock()
    bot.send_video = AsyncMock()
    gw = _gateway()
    tool_args: dict = {}
    await gw._handle_event(
        {"type": "tool_call", "id": "c1", "name": "generate_video",
         "arguments": {"prompt": "a pikachu surfing"}},
        AsyncMock(), bot, 42, tool_args,
    )
    await gw._handle_event(
        {"type": "tool_result", "id": "c1", "name": "generate_video",
         "ok": True, "content": str(clip)},
        AsyncMock(), bot, 42, tool_args,
    )
    assert bot.send_video.await_args.kwargs["caption"] == "a pikachu surfing"


async def test_handle_event_heartbeat_renders_generic_working_line():
    # A heartbeat tick (fraction < 0) must render as a plain "working…" line, NOT the video bar.
    editor = AsyncMock()
    await _gateway()._handle_event(
        {"type": "tool_progress", "name": "bash", "fraction": -1.0, "label": "1:05"},
        editor, Mock(), 42, {},
    )
    line = editor.progress.await_args.args[0]
    assert "🛠️" in line and "bash" in line and "1:05" in line
    assert "Generating video" not in line


def test_progress_bar_renders_fraction_and_clamps():
    assert _progress_bar(0.0, "0/40") == "🎬 Generating video ░░░░░░░░░░░░ 0% (0/40)"
    assert _progress_bar(0.5, "20/40").startswith("🎬 Generating video ██████░░░░░░ 50%")
    # out-of-range fractions must clamp, never overrun the bar or show >100%
    assert _progress_bar(1.5).startswith("🎬 Generating video ████████████ 100%")
    assert "-" not in _progress_bar(-0.2)


def test_clip_caption_caps_at_telegram_limit():
    assert _clip_caption(None) is None
    assert _clip_caption("  hi  ") == "hi"
    long = "x" * 2000
    out = _clip_caption(long)
    assert len(out) <= 1024 and out.endswith("…")
