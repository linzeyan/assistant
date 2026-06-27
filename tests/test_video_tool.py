"""generate_video tool driven through a fake backend (no real mlx-video here)."""

from __future__ import annotations

from pathlib import Path

from assistant.models.mlx_video import _valid_num_frames
from assistant.tools import build_registry
from assistant.tools.base import ToolContext


class FakeVideo:
    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[tuple] = []
        self.progress_seen = None

    def available(self) -> bool:
        return self._available

    async def generate_video(self, prompt, *, num_frames=None, seed=None, progress=None) -> Path:
        self.calls.append((prompt, num_frames, seed))
        self.progress_seen = progress
        if progress is not None:
            progress(0.5, "1/2")  # a backend would call this per denoising step
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


def test_valid_num_frames_rounds_to_4n_plus_1():
    # Wan/LTX assert (num_frames - 1) % 4 == 0; a free-form request must be coerced to a
    # valid count, not crash the whole generation on a bad number the model happened to pick.
    assert _valid_num_frames(81) == 81  # already valid (4*20 + 1) → unchanged
    assert _valid_num_frames(48) == 45  # rounds down to the nearest 4n+1
    assert _valid_num_frames(1) == 5  # clamped up to the minimum sane count
    for n in range(5, 200):
        assert (_valid_num_frames(n) - 1) % 4 == 0  # always lands on a valid count
