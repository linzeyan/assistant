import importlib.util
from pathlib import Path

import pytest

from assistant.images.mlx_backend import MlxImageBackend
from assistant.tools import build_registry
from assistant.tools.base import ToolContext


class _FakeImages:
    def __init__(self, available=True):
        self._available = available
        self.calls = []

    def available(self):
        return self._available

    async def generate_image(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return Path("/tmp/fake.png")


async def test_image_tool_unavailable_without_backend(tmp_path):
    tool = build_registry().get("generate_image")
    res = await tool.handler({"prompt": "a cat"}, ToolContext(cwd=tmp_path))
    assert res.ok is False and "unavailable" in res.content


async def test_image_tool_unavailable_when_backend_not_ready(tmp_path):
    tool = build_registry().get("generate_image")
    ctx = ToolContext(cwd=tmp_path, images=_FakeImages(available=False))
    res = await tool.handler({"prompt": "a cat"}, ctx)
    assert res.ok is False


async def test_image_tool_success_returns_path(tmp_path):
    fake = _FakeImages(available=True)
    tool = build_registry().get("generate_image")
    res = await tool.handler({"prompt": "a cat", "seed": 7}, ToolContext(cwd=tmp_path, images=fake))
    assert res.ok and res.content.endswith("fake.png")
    assert fake.calls[0][0] == "a cat" and fake.calls[0][1]["seed"] == 7


async def test_generate_image_passes_size_through(tmp_path):
    fake = _FakeImages(available=True)
    tool = build_registry().get("generate_image")
    res = await tool.handler(
        {"prompt": "a cat", "width": 512, "height": 768},
        ToolContext(cwd=tmp_path, images=fake),
    )
    assert res.ok
    assert fake.calls[0][1]["width"] == 512 and fake.calls[0][1]["height"] == 768


def test_mlx_backend_runtime_knobs(tmp_path):
    # The /imageset + GUI knobs are pure state (no mflux), so they're testable anywhere.
    backend = MlxImageBackend(tmp_path)
    assert backend.size == (1024, 1024)  # default
    backend.set_size("768")
    assert backend.size == (768, 768)
    backend.set_size("bogus")  # unknown preset is ignored, not an error
    assert backend.size == (768, 768)
    backend.set_steps(12)
    assert backend.steps == 12
    backend.set_steps(0)  # non-positive clears back to the per-alias default
    assert backend.steps is None
    backend.set_model("dev")
    assert backend.model == "dev"


def test_mlx_backend_seeds_defaults_from_config(tmp_path):
    backend = MlxImageBackend(tmp_path, width=512, height=512, steps=6)
    assert backend.size == (512, 512) and backend.steps == 6


def test_mlx_backend_available_reflects_mflux_presence(tmp_path):
    backend = MlxImageBackend(tmp_path)
    assert backend.available() == (importlib.util.find_spec("mflux") is not None)


async def test_mlx_backend_raises_when_unavailable(tmp_path):
    backend = MlxImageBackend(tmp_path)
    if backend.available():
        pytest.skip("mflux is installed in this environment")
    with pytest.raises(RuntimeError):
        await backend.generate_image("x")
