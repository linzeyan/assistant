"""generate_video tool driven through a fake backend (no real mlx-video here)."""

from __future__ import annotations

from pathlib import Path

from assistant.tools import build_registry
from assistant.tools.base import ToolContext


class FakeVideo:
    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return self._available

    async def generate_video(self, prompt, *, num_frames=None, seed=None) -> Path:
        self.calls.append((prompt, num_frames, seed))
        return Path("/tmp/vid_abc123.mp4")


def _tool():
    return build_registry().get("generate_video")


async def test_generate_video_returns_path(tmp_path):
    backend = FakeVideo()
    ctx = ToolContext(cwd=tmp_path, video=backend)
    res = await _tool().handler(
        {"prompt": "a dog surfing", "num_frames": 48, "seed": 7}, ctx
    )
    assert res.ok and res.content.endswith(".mp4")
    assert backend.calls == [("a dog surfing", 48, 7)]


async def test_generate_video_unavailable(tmp_path):
    ctx = ToolContext(cwd=tmp_path, video=FakeVideo(available=False))
    res = await _tool().handler({"prompt": "x"}, ctx)
    assert not res.ok and "unavailable" in res.content


async def test_generate_video_no_backend(tmp_path):
    ctx = ToolContext(cwd=tmp_path)  # video=None
    res = await _tool().handler({"prompt": "x"}, ctx)
    assert not res.ok and "unavailable" in res.content
