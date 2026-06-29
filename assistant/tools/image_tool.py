"""Image generation + editing tools.

On success the result content is the saved file path; the Telegram gateway and the
GUI special-case a successful ``generate_image`` / ``edit_image`` result to render the
actual image.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="generate_image",
    description="Generate an image from a text prompt and save it to disk.",
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string"},
            "steps": {"type": "integer", "description": "Inference steps (optional)."},
            "seed": {"type": "integer", "description": "Random seed (optional)."},
            "width": {"type": "integer", "description": "Output width in px (optional)."},
            "height": {"type": "integer", "description": "Output height in px (optional)."},
        },
        "required": ["prompt"],
    },
)
async def generate_image(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.images is None or not ctx.images.available():
        return ToolResult(
            False,
            "image generation is unavailable (install mflux on Apple Silicon).",
        )
    try:
        path = await ctx.images.generate_image(
            args["prompt"],
            steps=args.get("steps"),
            seed=args.get("seed"),
            width=args.get("width"),
            height=args.get("height"),
        )
    except Exception as exc:
        return ToolResult(False, f"image generation failed: {exc}")
    return ToolResult(True, str(path))


@registry.tool(
    name="edit_image",
    description=(
        "Edit or transform existing image(s) with a text instruction (image-to-image, "
        "e.g. Qwen-Image-Edit). Provide one input image, or several for multi-reference."
    ),
    parameters={
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The edit instruction."},
            "image_path": {
                "type": "string",
                "description": "Path to the input image to edit.",
            },
            "image_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Multiple input images, for multi-reference edits.",
            },
            "steps": {"type": "integer", "description": "Inference steps (optional)."},
            "seed": {"type": "integer", "description": "Random seed (optional)."},
            "guidance": {
                "type": "number",
                "description": "Guidance scale — how strictly to follow the instruction (optional).",
            },
        },
        "required": ["prompt"],
    },
)
async def edit_image(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.images is None or not ctx.images.available():
        return ToolResult(
            False, "image editing is unavailable (install mflux>=0.18 on Apple Silicon)."
        )
    # Accept either a single image_path or a list; normalise to a resolved list.
    raw = list(args.get("image_paths") or [])
    if not raw and args.get("image_path"):
        raw = [args["image_path"]]
    if not raw:
        return ToolResult(False, "edit_image requires image_path or image_paths.")
    paths: list[str] = []
    for item in raw:
        p = Path(item)
        if not p.is_absolute():
            p = ctx.cwd / p
        if not p.is_file():
            return ToolResult(False, f"image not found: {item}")
        paths.append(str(p))
    try:
        out = await ctx.images.edit_image(
            args["prompt"],
            paths,
            steps=args.get("steps"),
            seed=args.get("seed"),
            guidance=args.get("guidance"),
        )
    except Exception as exc:
        return ToolResult(False, f"image edit failed: {exc}")
    return ToolResult(True, str(out))
