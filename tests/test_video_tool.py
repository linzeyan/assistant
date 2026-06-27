"""generate_video tool driven through a fake backend (no real mlx-video here)."""

from __future__ import annotations

from pathlib import Path

from assistant.models.mlx_video import DEFAULT_RESOLUTION, _RESOLUTIONS, _valid_num_frames
from assistant.tools import build_registry
from assistant.tools.base import ToolContext


class FakeVideo:
    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[dict] = []
        self.progress_seen = None

    def available(self) -> bool:
        return self._available

    async def generate_video(
        self,
        prompt,
        *,
        resolution=None,
        num_frames=None,
        steps=None,
        seed=None,
        negative_prompt=None,
        progress=None,
    ) -> Path:
        self.calls.append(
            {
                "prompt": prompt,
                "resolution": resolution,
                "num_frames": num_frames,
                "steps": steps,
                "seed": seed,
                "negative_prompt": negative_prompt,
            }
        )
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
    call = backend.calls[0]
    assert (call["prompt"], call["num_frames"], call["seed"]) == ("a dog surfing", 48, 7)
    # Unspecified knobs reach the backend as None, so it applies its own defaults (360p etc.).
    assert call["resolution"] is None and call["steps"] is None


async def test_generate_video_forwards_overrides(tmp_path):
    # WHY: the model maps a user's "720p, 5 seconds, quick draft" onto these args; the tool
    # must pass them through verbatim rather than swallow them.
    backend = FakeVideo()
    ctx = ToolContext(cwd=tmp_path, video=backend)
    await _tool().handler(
        {"prompt": "x", "resolution": "720p", "steps": 20, "negative_prompt": "blurry"}, ctx
    )
    call = backend.calls[0]
    assert call["resolution"] == "720p"
    assert call["steps"] == 20
    assert call["negative_prompt"] == "blurry"


async def test_generate_video_unavailable(tmp_path):
    ctx = ToolContext(cwd=tmp_path, video=FakeVideo(available=False))
    res = await _tool().handler({"prompt": "x"}, ctx)
    assert not res.ok and "unavailable" in res.content


async def test_generate_video_no_backend(tmp_path):
    ctx = ToolContext(cwd=tmp_path)  # video=None
    res = await _tool().handler({"prompt": "x"}, ctx)
    assert not res.ok and "unavailable" in res.content


def test_resolutions_are_32_aligned_and_default_is_cheapest():
    # WHY: Wan's VAE stride (16) × patch (2) requires width/height divisible by 32, and the
    # default must be the smallest so a clip is fast (a few min) unless the user opts up.
    assert DEFAULT_RESOLUTION in _RESOLUTIONS
    for name, (w, h) in _RESOLUTIONS.items():
        assert w % 32 == 0 and h % 32 == 0, name
    areas = {n: w * h for n, (w, h) in _RESOLUTIONS.items()}
    assert areas[DEFAULT_RESOLUTION] == min(areas.values())


def test_valid_num_frames_rounds_to_4n_plus_1():
    # Wan/LTX assert (num_frames - 1) % 4 == 0; a free-form request must be coerced to a
    # valid count, not crash the whole generation on a bad number the model happened to pick.
    assert _valid_num_frames(81) == 81  # already valid (4*20 + 1) → unchanged
    assert _valid_num_frames(48) == 45  # rounds down to the nearest 4n+1
    assert _valid_num_frames(1) == 5  # clamped up to the minimum sane count
    for n in range(5, 200):
        assert (_valid_num_frames(n) - 1) % 4 == 0  # always lands on a valid count
