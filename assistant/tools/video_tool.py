"""Video generation tool.

Like ``generate_image``, a successful result's content is the saved file path; the
Telegram gateway / GUI can special-case it to play the clip back.
"""

from __future__ import annotations

from .base import ToolContext, ToolResult, service_available
from .registry import registry


@registry.tool(
    name="generate_video",
    description=(
        "Generate a short video from a text prompt and save it to disk. All settings have "
        "sensible defaults — only pass the optional ones when the user explicitly asks (e.g. "
        "a different resolution, length, or quality)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "What the video should depict (in English works best).",
            },
            "resolution": {
                "type": "string",
                "enum": ["360p", "480p", "540p", "720p"],
                "description": (
                    "Output resolution. Defaults to 360p, which is fast (a few minutes); "
                    "higher is sharper but much slower (720p ≈ 20 min). Only raise it if the "
                    "user asks for higher quality / a specific resolution."
                ),
            },
            "num_frames": {
                "type": "integer",
                "description": (
                    "Clip length in frames at 24 fps (must be 4n+1; auto-corrected). Default "
                    "≈81 (~3.4s). Use ~121 for ~5s, ~241 for ~10s."
                ),
            },
            "steps": {
                "type": "integer",
                "description": (
                    "Denoising steps (default 40). Fewer is faster but lower quality; lower "
                    "it (e.g. 20-25) only when the user wants a quick draft."
                ),
            },
            "seed": {"type": "integer", "description": "Random seed for a reproducible clip."},
            "negative_prompt": {
                "type": "string",
                "description": "Things to avoid in the video (optional).",
            },
        },
        "required": ["prompt"],
    },
    check_fn=service_available("video"),
)
async def generate_video(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.video is None or not ctx.video.available():
        return ToolResult(
            False, "video generation is unavailable (install mlx-video on Apple Silicon)."
        )
    try:
        path = await ctx.video.generate_video(
            args["prompt"],
            resolution=args.get("resolution"),
            num_frames=args.get("num_frames"),
            steps=args.get("steps"),
            seed=args.get("seed"),
            negative_prompt=args.get("negative_prompt"),
            progress=ctx.on_progress,  # streams a per-step progress bar (None = no consumer)
        )
    except Exception as exc:
        return ToolResult(False, f"video generation failed: {exc}")
    return ToolResult(True, str(path))
