"""MLX image generation + editing backend via mflux (FLUX / Qwen-Image-Edit on Apple
Silicon).

mflux is an optional, heavy, Apple-Silicon-only dependency, so this backend is
defensive: ``available()`` reports whether mflux is importable, and inference is
offloaded to a worker thread because MLX is synchronous and blocking — running it on
the event loop would stall chat streaming and the Telegram gateway (plan Part E, risk 2).

Targets mflux >= 0.18 (text-to-image via Flux1, image editing via QwenImageEdit). The
mflux API has moved across versions (top-level ``Config``/``Flux1`` were removed and
``generate_image`` now takes flat kwargs instead of a ``Config`` object); all imports are
isolated in the sync methods so a version bump touches only those.
"""

from __future__ import annotations

import asyncio
import importlib.util
import shutil
import subprocess
import uuid
from pathlib import Path

from .service import MediaService

# Named square sizes offered by the /imageset picker and the GUI. The tool still accepts an
# arbitrary width/height; these are just the one-tap presets (FLUX trains at 1024).
IMAGE_SIZES: dict[str, tuple[int, int]] = {
    "512": (512, 512),
    "768": (768, 768),
    "1024": (1024, 1024),
}
DEFAULT_IMAGE_SIZE = "512"  # lighter/faster default; bump per request or via /imageset
# mflux text-to-image aliases. Empty by default: the schnell/dev aliases pull a multi-GB FLUX.1
# checkpoint from HuggingFace on first use (looks frozen) and the user has fast local mlx-gen
# models, so the /image picker offers only on-disk checkpoints (discover_image_checkpoints) that
# route to the mlxgen CLI. The mflux in-venv path below stays as a fallback if a model is ever
# pointed at an mflux alias directly.
IMAGE_MODELS: tuple[str, ...] = ()

# mlx-gen models generate via the `mlxgen` CLI (a uv-tool install), not mflux in-venv.
_MLXGEN_TIMEOUT_S = 1800  # a large 8-bit checkpoint can take many minutes per image


def _mlxgen_exe() -> str | None:
    """Resolve the mlxgen binary. The GUI-spawned backend's PATH frequently lacks ~/.local/bin
    (the uv-tool shim dir), so fall back to the conventional install path explicitly rather than
    relying on PATH alone."""
    exe = shutil.which("mlxgen")
    if exe:
        return exe
    fallback = Path.home() / ".local" / "bin" / "mlxgen"
    return str(fallback) if fallback.exists() else None


class MlxImageBackend(MediaService):
    def __init__(
        self,
        images_dir: Path,
        model: str = "",  # resolved at startup to a local image checkpoint (see main.py)
        *,
        edit_quantize: int | None = None,
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
    ):
        self._dir = Path(images_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        # Optional quantization for the large Qwen-Image-Edit model (8/4) to fit tighter
        # unified memory; None = full precision.
        self._edit_quantize = edit_quantize
        # Runtime-switchable defaults (Telegram /imageset, GUI). A per-request tool arg still
        # overrides these; steps None lets _generate_sync pick a per-alias default.
        self._width = width or IMAGE_SIZES[DEFAULT_IMAGE_SIZE][0]
        self._height = height or IMAGE_SIZES[DEFAULT_IMAGE_SIZE][1]
        self._steps = steps if steps and steps > 0 else None

    @property
    def model(self) -> str:
        return self._model

    def set_model(self, name: str) -> None:
        if name:
            self._model = name

    @property
    def size(self) -> tuple[int, int]:
        return (self._width, self._height)

    def set_size(self, name: str) -> None:
        """Set the default output size from a named preset (see IMAGE_SIZES)."""
        if name in IMAGE_SIZES:
            self._width, self._height = IMAGE_SIZES[name]

    @property
    def steps(self) -> int | None:
        return self._steps

    def set_steps(self, steps: int | None) -> None:
        self._steps = steps if steps and steps > 0 else None

    def available(self) -> bool:
        # Either backend counts: mflux (in-venv) serves the schnell/dev aliases, the mlxgen CLI
        # serves on-disk mlx-gen checkpoints. /image stays usable if only one is installed.
        return importlib.util.find_spec("mflux") is not None or _mlxgen_exe() is not None

    def _is_mlxgen_model(self) -> bool:
        # A disk-path model is an mlx-gen checkpoint → route to the mlxgen CLI. mflux's in-venv
        # loader can't read these; aliases (schnell/dev) are not directories, so they stay on
        # the mflux path.
        return Path(self._model).is_dir()

    def _build_mlxgen_cmd(
        self,
        exe: str,
        out: Path,
        prompt: str,
        *,
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
        image_paths: list[str] | None,
    ) -> list[str]:
        """Assemble the `mlxgen generate` argv (factored out so it's unit-testable without
        actually invoking the CLI). Mirrors the proven invocations in the mlx-dir Makefile."""
        cmd = [
            exe, "generate",
            "--model", self._model,
            "--prompt", prompt,
            "--width", str(width or self._width),
            "--height", str(height or self._height),
            "--output", str(out),
        ]
        eff_steps = steps or self._steps
        if eff_steps:
            cmd += ["--steps", str(eff_steps)]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        # img2img / edit input. --image-path is mlxgen's single-image compat flag (what the
        # Makefile uses); --image repeats for multi-reference edits.
        if image_paths:
            if len(image_paths) == 1:
                cmd += ["--image-path", image_paths[0]]
            else:
                for p in image_paths:
                    cmd += ["--image", p]
        return cmd

    def _mlxgen_generate(
        self,
        prompt: str,
        *,
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
        image_paths: list[str] | None,
    ) -> Path:
        exe = _mlxgen_exe()
        if exe is None:
            raise RuntimeError(
                "mlx-gen CLI not found; install it with: uv tool install mlx-gen"
            )
        out = self._dir / f"{'edit' if image_paths else 'img'}_{uuid.uuid4().hex[:8]}.png"
        cmd = self._build_mlxgen_cmd(
            exe, out, prompt, steps=steps, seed=seed,
            width=width, height=height, image_paths=image_paths,
        )
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_MLXGEN_TIMEOUT_S, check=False
        )
        # mlxgen returns 0 even when it only printed help, so also require the file to exist.
        if proc.returncode != 0 or not out.is_file():
            tail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise RuntimeError(f"mlxgen generate failed (exit {proc.returncode}): {tail}")
        return out

    async def generate_image(
        self,
        prompt: str,
        *,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "image generation requires mflux. Install it with: pip install mflux"
            )
        if not self._model:
            raise RuntimeError(
                "no image model selected; pick one with /image (Telegram) or place an mlx-gen "
                "checkpoint in your model dirs."
            )
        return await asyncio.to_thread(
            self._generate_sync, prompt, steps, seed, width, height
        )

    def _generate_sync(
        self,
        prompt: str,
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
    ) -> Path:
        if self._is_mlxgen_model():
            return self._mlxgen_generate(
                prompt, steps=steps, seed=seed, width=width, height=height, image_paths=None
            )
        # Lazy, isolated import (mflux >= 0.18 layout): Flux1 lives under the txt2img
        # variant package and generate_image takes flat kwargs (no Config object).
        from mflux.models.flux.variants.txt2img.flux import Flux1

        flux = Flux1.from_name(self._model)
        # Precedence: explicit per-request arg > the /imageset default > a per-alias fallback.
        eff_steps = steps or self._steps or (4 if self._model == "schnell" else 20)
        image = flux.generate_image(
            seed=seed if seed is not None else 0,
            prompt=prompt,
            num_inference_steps=eff_steps,
            height=height or self._height,
            width=width or self._width,
        )
        out = self._dir / f"img_{uuid.uuid4().hex[:8]}.png"
        image.save(path=str(out))
        return out

    async def edit_image(
        self,
        prompt: str,
        image_paths: list[str],
        *,
        steps: int | None = None,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
        guidance: float | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "image editing requires mflux>=0.18 (uv pip install -U mflux)"
            )
        if not self._model:
            raise RuntimeError(
                "no image model selected; pick an edit-capable checkpoint with /image (Telegram)."
            )
        return await asyncio.to_thread(
            self._edit_sync, prompt, list(image_paths), steps, seed, width, height, guidance
        )

    def _edit_sync(
        self,
        prompt: str,
        image_paths: list[str],
        steps: int | None,
        seed: int | None,
        width: int | None,
        height: int | None,
        guidance: float | None,
    ) -> Path:
        if self._is_mlxgen_model():
            # mlxgen auto-routes to the edit/img2img path from the presence of --image-path.
            return self._mlxgen_generate(
                prompt, steps=steps, seed=seed, width=width, height=height,
                image_paths=image_paths,
            )
        # Qwen-Image-Edit via mflux >= 0.18. The qwen edit module is absent on older mflux,
        # so an ImportError here is surfaced as a clear "upgrade mflux" rather than a crash.
        try:
            from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
        except ImportError as exc:
            raise RuntimeError(
                "this mflux build has no Qwen image edit; upgrade to mflux>=0.18 "
                "(Settings > Managed tools > 更新套件) and restart the backend."
            ) from exc

        editor = QwenImageEdit(quantize=self._edit_quantize)
        # Default to 8 steps / guidance 2.5 (mflux's default of 4 is Lightning-LoRA tuned
        # and looks broken for a normal edit); height/width default to the input's size.
        kwargs: dict = {
            "seed": seed if seed is not None else 0,
            "prompt": prompt,
            "image_paths": image_paths,
            "num_inference_steps": steps or 8,
            "guidance": guidance if guidance is not None else 2.5,
        }
        if height:
            kwargs["height"] = height
        if width:
            kwargs["width"] = width
        image = editor.generate_image(**kwargs)
        out = self._dir / f"edit_{uuid.uuid4().hex[:8]}.png"
        image.save(path=str(out))
        return out
