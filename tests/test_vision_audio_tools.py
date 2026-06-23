"""Vision + audio tools driven through fake backends (no real MLX models here)."""

from __future__ import annotations

from pathlib import Path

from assistant.tools import build_registry
from assistant.tools.base import ToolContext


class FakeVision:
    def __init__(self, available: bool = True, text: str = "a red square"):
        self._available = available
        self._text = text

    def available(self) -> bool:
        return self._available

    async def describe(self, image_paths, prompt) -> str:
        return self._text


class FakeAudio:
    def __init__(self, available: bool = True, text: str = "hello world"):
        self._available = available
        self._text = text

    def available(self) -> bool:
        return self._available

    async def transcribe(self, audio_path) -> str:
        return self._text

    async def speak(self, text) -> Path:
        return Path("/tmp/tts_out.wav")


def _tool(name):
    return build_registry().get(name)


async def test_view_image_returns_description(tmp_path):
    (tmp_path / "x.png").write_bytes(b"fakepng")
    ctx = ToolContext(cwd=tmp_path, vision=FakeVision(text="a cat on a mat"))
    res = await _tool("view_image").handler({"path": "x.png", "question": "what?"}, ctx)
    assert res.ok and res.content == "a cat on a mat"


async def test_view_image_missing_file(tmp_path):
    ctx = ToolContext(cwd=tmp_path, vision=FakeVision())
    res = await _tool("view_image").handler({"path": "nope.png"}, ctx)
    assert not res.ok and "not found" in res.content


async def test_view_image_unavailable(tmp_path):
    (tmp_path / "x.png").write_bytes(b"p")
    ctx = ToolContext(cwd=tmp_path, vision=FakeVision(available=False))
    res = await _tool("view_image").handler({"path": "x.png"}, ctx)
    assert not res.ok and "unavailable" in res.content


async def test_transcribe_audio_returns_text(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    ctx = ToolContext(cwd=tmp_path, audio=FakeAudio(text="hi there"))
    res = await _tool("transcribe_audio").handler({"path": "a.wav"}, ctx)
    assert res.ok and res.content == "hi there"


async def test_text_to_speech_returns_audio_path(tmp_path):
    ctx = ToolContext(cwd=tmp_path, audio=FakeAudio())
    res = await _tool("text_to_speech").handler({"text": "speak this"}, ctx)
    assert res.ok and res.content.endswith(".wav")


async def test_audio_tools_unavailable(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    ctx = ToolContext(cwd=tmp_path, audio=FakeAudio(available=False))
    assert not (await _tool("transcribe_audio").handler({"path": "a.wav"}, ctx)).ok
    assert not (await _tool("text_to_speech").handler({"text": "x"}, ctx)).ok
