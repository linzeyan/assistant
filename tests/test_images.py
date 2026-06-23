"""Image edit tool plumbing (the mflux-touching backend is exercised live, not here)."""

from __future__ import annotations

from pathlib import Path

from assistant.images.service import MediaService
from assistant.tools.base import ToolContext
from assistant.tools.image_tool import edit_image


class FakeImages(MediaService):
    """A MediaService that records calls instead of running mflux."""

    def __init__(self, available: bool = True):
        self._available = available
        self.calls: list[tuple] = []

    def available(self) -> bool:
        return self._available

    async def generate_image(self, prompt, *, steps=None, seed=None, width=None, height=None):
        self.calls.append(("gen", prompt))
        return Path("/tmp/out.png")

    async def edit_image(
        self, prompt, image_paths, *, steps=None, seed=None, width=None, height=None, guidance=None
    ):
        self.calls.append(("edit", prompt, list(image_paths)))
        return Path("/tmp/edited.png")


async def test_edit_image_resolves_relative_path_and_calls_backend(tmp_path):
    img = tmp_path / "in.png"
    img.write_bytes(b"x")
    fake = FakeImages()
    ctx = ToolContext(cwd=tmp_path, images=fake)
    res = await edit_image({"prompt": "make it blue", "image_path": "in.png"}, ctx)
    assert res.ok and res.content == "/tmp/edited.png"
    # Relative path resolved against cwd before reaching the backend.
    assert fake.calls == [("edit", "make it blue", [str(img)])]


async def test_edit_image_multi_reference(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    fake = FakeImages()
    ctx = ToolContext(cwd=tmp_path, images=fake)
    res = await edit_image({"prompt": "blend", "image_paths": ["a.png", "b.png"]}, ctx)
    assert res.ok
    assert fake.calls[0] == ("edit", "blend", [str(a), str(b)])


async def test_edit_image_missing_file_is_a_clean_error(tmp_path):
    fake = FakeImages()
    ctx = ToolContext(cwd=tmp_path, images=fake)
    res = await edit_image({"prompt": "x", "image_path": "nope.png"}, ctx)
    assert not res.ok and "not found" in res.content
    assert fake.calls == []  # never reached the backend


async def test_edit_image_requires_an_input_image(tmp_path):
    ctx = ToolContext(cwd=tmp_path, images=FakeImages())
    res = await edit_image({"prompt": "x"}, ctx)
    assert not res.ok and "requires" in res.content


async def test_edit_image_unavailable_backend(tmp_path):
    ctx = ToolContext(cwd=tmp_path, images=FakeImages(available=False))
    res = await edit_image({"prompt": "x", "image_path": "in.png"}, ctx)
    assert not res.ok and "unavailable" in res.content
