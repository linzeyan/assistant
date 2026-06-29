import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

import assistant.images.mlx_backend as mlx_backend
from assistant.images.mlx_backend import MlxImageBackend, _mlxgen_exe
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
    assert backend.size == (512, 512)  # default (lighter/faster)
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


def test_mlx_backend_available_reflects_either_backend(tmp_path):
    # available() is True if EITHER mflux (in-venv, for schnell/dev) or the mlxgen CLI (for
    # on-disk mlx-gen checkpoints) is present, so /image stays usable with just one installed.
    backend = MlxImageBackend(tmp_path)
    expected = importlib.util.find_spec("mflux") is not None or _mlxgen_exe() is not None
    assert backend.available() == expected


async def test_mlx_backend_raises_when_unavailable(tmp_path):
    backend = MlxImageBackend(tmp_path)
    if backend.available():
        pytest.skip("mflux/mlxgen is installed in this environment")
    with pytest.raises(RuntimeError):
        await backend.generate_image("x")


def test_is_mlxgen_model_only_for_disk_path(tmp_path):
    backend = MlxImageBackend(tmp_path / "out")
    backend.set_model("schnell")
    assert backend._is_mlxgen_model() is False  # an mflux alias is not a directory
    disk = tmp_path / "z-image-turbo-8bit"
    disk.mkdir()
    backend.set_model(str(disk))
    assert backend._is_mlxgen_model() is True


def test_build_mlxgen_cmd_txt2img_and_edit(tmp_path):
    backend = MlxImageBackend(tmp_path, width=1024, height=1024)
    backend.set_model("/models/z-image")
    out = tmp_path / "o.png"
    cmd = backend._build_mlxgen_cmd(
        "mlxgen", out, "hi", steps=6, seed=1, width=None, height=None, image_paths=None
    )
    assert cmd == [
        "mlxgen", "generate", "--model", "/models/z-image", "--prompt", "hi",
        "--width", "1024", "--height", "1024", "--output", str(out),
        "--steps", "6", "--seed", "1",
    ]
    # A single edit image uses --image-path (mlxgen's compat flag, as in the Makefile); steps
    # default off when unset.
    edit1 = backend._build_mlxgen_cmd(
        "mlxgen", out, "edit", steps=None, seed=None, width=None, height=None,
        image_paths=["/a.png"],
    )
    assert "--image-path" in edit1 and "/a.png" in edit1 and "--steps" not in edit1
    # Multiple images repeat --image (multi-reference edit).
    edit2 = backend._build_mlxgen_cmd(
        "mlxgen", out, "edit", steps=None, seed=None, width=None, height=None,
        image_paths=["/a.png", "/b.png"],
    )
    assert edit2.count("--image") == 2


async def test_generate_routes_to_mlxgen_for_disk_model(tmp_path, monkeypatch):
    model_dir = tmp_path / "z-image-turbo-8bit"
    model_dir.mkdir()
    backend = MlxImageBackend(tmp_path / "out")
    backend.set_model(str(model_dir))
    monkeypatch.setattr(mlx_backend, "_mlxgen_exe", lambda: "/usr/bin/mlxgen")
    captured: dict = {}

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        out = Path(cmd[cmd.index("--output") + 1])
        out.write_bytes(b"\x89PNG")  # pretend an image was produced
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mlx_backend.subprocess, "run", fake_run)
    p = await backend.generate_image("a doraemon", steps=6, seed=3, width=768, height=768)
    assert p.is_file()
    cmd = captured["cmd"]
    assert cmd[:2] == ["/usr/bin/mlxgen", "generate"]
    assert str(model_dir) in cmd
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--width") + 1] == "768"


async def test_mlxgen_generate_raises_on_failure(tmp_path, monkeypatch):
    model_dir = tmp_path / "z-image-turbo-8bit"
    model_dir.mkdir()
    backend = MlxImageBackend(tmp_path / "out")
    backend.set_model(str(model_dir))
    monkeypatch.setattr(mlx_backend, "_mlxgen_exe", lambda: "/usr/bin/mlxgen")
    monkeypatch.setattr(
        mlx_backend.subprocess, "run",
        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr="boom"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        await backend.generate_image("x")
