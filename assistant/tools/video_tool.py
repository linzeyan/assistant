"""Video generation tool.

Like ``generate_image``, a successful result's content is the saved file path; the
Telegram gateway / GUI can special-case it to play the clip back.
"""

from __future__ import annotations

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="generate_video",
    description="Generate a short video from a text prompt and save it to disk.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "num_frames": {
                "type": "integer",
                "description": "Number of frames to generate (optional).",
            },
            "seed": {"type": "integer", "description": "Random seed (optional)."},
        },
        "required": ["prompt"],
    },
)
async def generate_video(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.video is None or not ctx.video.available():
        return ToolResult(
            False, "video generation is unavailable (install mlx-video on Apple Silicon)."
        )
    try:
        path = await ctx.video.generate_video(
            args["prompt"], num_frames=args.get("num_frames"), seed=args.get("seed")
        )
    except Exception as exc:
        return ToolResult(False, f"video generation failed: {exc}")
    return ToolResult(True, str(path))
