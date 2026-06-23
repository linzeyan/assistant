"""Vision tool: let the (text) agent read an image via a local VLM."""

from __future__ import annotations

from pathlib import Path

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="view_image",
    description="Look at an image file and answer a question about it (vision model).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the image file."},
            "question": {
                "type": "string",
                "description": "What to ask about the image (optional).",
            },
        },
        "required": ["path"],
    },
)
async def view_image(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.vision is None or not ctx.vision.available():
        return ToolResult(
            False, "vision is unavailable (install mlx-vlm on Apple Silicon)."
        )
    raw = args["path"]
    path = Path(raw)
    if not path.is_absolute():
        path = ctx.cwd / path
    if not path.is_file():
        return ToolResult(False, f"image not found: {raw}")
    question = args.get("question") or "Describe this image in detail."
    try:
        text = await ctx.vision.describe([str(path)], question)
    except Exception as exc:
        return ToolResult(False, f"vision failed: {exc}")
    return ToolResult(True, text)
