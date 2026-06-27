"""Native text-to-video backend via Blaizzy's mlx-video (Wan / LTX on MLX).

Roadmap Phase 6: local video generation. Optional and defensive like the other MLX
backends — ``available()`` reports importability and generation runs in a worker
thread. Video is the heaviest MLX workload, so keeping it off the event loop matters
most here (plan Part E, risk 2).

The real package is Blaizzy's ``mlx-video`` (installed from git, NOT the identically
named PyPI I/O library): generation lives at
``mlx_video.models.{wan_2,ltx_2}.generate.generate_video`` and loads weights from a
local *converted-MLX* checkpoint dir passed as ``model_dir`` (e.g. ``Wan2.2-TI2V-5B-mlx``).
The pipeline is chosen by name: ``"wan"`` (``wan_2``) or ``"ltx"`` (``ltx_2``).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path

# (fraction in [0, 1], short "step/total" label) — reported per denoising step.
ProgressFn = Callable[[float, str], None]

# Named resolutions → (width, height). Wan's VAE stride (16) × patch (2) means dims must be
# multiples of 32 (the lib floors to that anyway); these are 32-aligned ~16:9 sizes. Default
# is the smallest: TI2V-5B at 1280×704 takes ~20 min/clip on a Mac, but cost scales with
# pixel area, so 360p (~4× fewer pixels) brings a short clip down to a few minutes.
_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "360p": (640, 352),
    "480p": (832, 480),
    "540p": (960, 544),
    "720p": (1280, 704),
}
DEFAULT_RESOLUTION = "360p"


def _valid_num_frames(n: int) -> int:
    """Coerce a frame count to the Wan/LTX-required ``4·k + 1`` (the pipeline asserts this).

    The tool exposes a free integer and local models pick arbitrary values, so round down to
    the nearest valid count (min 5) instead of letting generation hard-fail on a bad request.
    """
    n = max(5, n)
    return n - ((n - 1) % 4)


@contextlib.contextmanager
def _instrument_denoise_progress(genmod, progress: ProgressFn | None) -> Iterator[None]:
    """Temporarily replace ``genmod.tqdm`` so the diffusion loop reports per-step progress.

    Blaizzy's generate iterates ``tqdm(range(steps), desc="Diffusion")`` with no callback
    seam, so this is the least-invasive way to surface progress: wrap only that bar (matched
    by ``desc``) and delegate every other tqdm call through unchanged. ``progress`` is invoked
    from the generation worker thread, so the agent loop hands us a thread-safe sink. The
    swap is process-global, but video generation is serialized (one heavy job at a time), and
    the original is always restored on exit.
    """
    if progress is None:
        yield
        return

    orig = genmod.tqdm

    def _patched(iterable=None, *args, **kwargs):
        if iterable is not None and kwargs.get("desc") == "Diffusion":
            steps = list(iterable)
            total = len(steps) or 1

            def _stream():
                for done, item in enumerate(steps):
                    yield item  # the lib runs one denoising step for this item
                    try:
                        progress((done + 1) / total, f"{done + 1}/{total}")
                    except Exception:
                        pass  # a progress-sink failure must never break generation

            return _stream()
        return orig(iterable, *args, **kwargs)

    genmod.tqdm = _patched
    try:
        yield
    finally:
        genmod.tqdm = orig


class VideoService(ABC):
    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        *,
        resolution: str | None = None,
        num_frames: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        progress: ProgressFn | None = None,
    ) -> Path:
        """Generate a short video and return the path to the saved file.

        ``progress``, when given, is called per denoising step with (fraction, label) so a
        caller (the agent loop) can stream a progress bar for this minutes-long workload."""


class MlxVideoBackend(VideoService):
    def __init__(
        self, video_dir: Path, model: str = "wan", *, checkpoint: Path | None = None
    ):
        self._dir = Path(video_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        # Local converted-MLX checkpoint dir handed to mlx-video as ``model_dir``. Unlike
        # image gen (mflux resolves an alias internally), Wan/LTX need an explicit on-disk
        # checkpoint — without it generation can't run. Runtime-switchable via set_checkpoint
        # so the Telegram /video picker (N28) can change it without a restart.
        self._checkpoint = Path(checkpoint) if checkpoint else None
        # Default generation knobs, runtime-switchable via the Telegram /videoset buttons.
        # A per-request arg (from the model's tool call) still overrides these; None steps
        # means "fall back to the checkpoint config" (≈40).
        self._resolution = DEFAULT_RESOLUTION
        self._steps: int | None = None

    @property
    def checkpoint(self) -> Path | None:
        return self._checkpoint

    def set_checkpoint(self, checkpoint: Path | None) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else None

    @property
    def resolution(self) -> str:
        return self._resolution

    def set_resolution(self, name: str) -> None:
        if name.lower() in _RESOLUTIONS:
            self._resolution = name.lower()

    @property
    def steps(self) -> int | None:
        return self._steps

    def set_steps(self, steps: int | None) -> None:
        self._steps = steps if steps is None or steps > 0 else None

    def available(self) -> bool:
        # Check the REAL generation submodule, not just top-level ``mlx_video``: the unrelated
        # PyPI "mlx-video" I/O lib also imports as ``mlx_video`` but has no ``models.wan_2``,
        # so a top-level check falsely reports availability against the wrong package.
        try:
            return (
                importlib.util.find_spec("mlx_video.models.wan_2.generate") is not None
            )
        except ImportError:
            return False

    async def generate_video(
        self,
        prompt: str,
        *,
        resolution: str | None = None,
        num_frames: int | None = None,
        steps: int | None = None,
        seed: int | None = None,
        negative_prompt: str | None = None,
        progress: ProgressFn | None = None,
    ) -> Path:
        if not self.available():
            raise RuntimeError(
                "video requires Blaizzy's mlx-video. "
                'Install with: uv pip install -e ".[video]"'
            )
        return await asyncio.to_thread(
            self._generate_sync,
            prompt,
            resolution,
            num_frames,
            steps,
            seed,
            negative_prompt,
            progress,
        )

    def _generate_sync(
        self,
        prompt: str,
        resolution: str | None,
        num_frames: int | None,
        steps: int | None,
        seed: int | None,
        negative_prompt: str | None,
        progress: ProgressFn | None = None,
    ) -> Path:
        if self._checkpoint is None or not self._checkpoint.is_dir():
            raise RuntimeError(
                "video generation needs a local converted-MLX checkpoint; set "
                "`video_checkpoint` to a Wan/LTX model dir (e.g. .../Wan2.2-TI2V-5B-mlx)."
            )
        # Single integration point with mlx-video: "wan" -> models.wan_2, "ltx" -> models.ltx_2.
        # Import the MODULE (not just the function) so we can instrument its denoising bar.
        if self._model == "ltx":
            from mlx_video.models.ltx_2 import generate as _genmod
        else:
            from mlx_video.models.wan_2 import generate as _genmod

        out = self._dir / f"vid_{uuid.uuid4().hex[:8]}.mp4"
        # Precedence: explicit per-request arg > the /videoset default > module default.
        # Everything left as None lets the lib fall back to the checkpoint's config defaults.
        width, height = _RESOLUTIONS.get(
            (resolution or self._resolution).lower(), _RESOLUTIONS[DEFAULT_RESOLUTION]
        )
        eff_steps = steps if steps is not None else self._steps
        kwargs: dict = {"width": width, "height": height}
        if num_frames is not None:
            kwargs["num_frames"] = _valid_num_frames(num_frames)
        if eff_steps is not None:
            kwargs["steps"] = eff_steps
        if seed is not None:
            kwargs["seed"] = seed
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        # generate_video writes to output_path and returns None — return the path we chose.
        # The lib exposes no progress callback, so to stream a per-step bar we temporarily
        # wrap the module-level tqdm it iterates over the diffusion steps with (desc=
        # "Diffusion"). Restored in finally so we never leave the lib's tqdm patched.
        with _instrument_denoise_progress(_genmod, progress):
            _genmod.generate_video(
                model_dir=str(self._checkpoint),
                prompt=prompt,
                output_path=str(out),
                **kwargs,
            )
        return out
