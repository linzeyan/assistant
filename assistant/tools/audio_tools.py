"""Audio tools: speech-to-text and text-to-speech via a local mlx-audio backend.

``text_to_speech`` returns the saved audio path as its content, mirroring
``generate_image`` — the Telegram gateway / GUI can special-case it to play back the
file.
"""

from __future__ import annotations

from pathlib import Path

from .base import ToolContext, ToolResult
from .registry import registry


@registry.tool(
    name="transcribe_audio",
    description="Transcribe an audio file to text (speech-to-text).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the audio file."},
        },
        "required": ["path"],
    },
)
async def transcribe_audio(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.audio is None or not ctx.audio.available():
        return ToolResult(
            False, "audio is unavailable (install mlx-audio on Apple Silicon)."
        )
    raw = args["path"]
    path = Path(raw)
    if not path.is_absolute():
        path = ctx.cwd / path
    if not path.is_file():
        return ToolResult(False, f"audio not found: {raw}")
    try:
        text = await ctx.audio.transcribe(str(path))
    except Exception as exc:
        return ToolResult(False, f"transcription failed: {exc}")
    return ToolResult(True, text)


@registry.tool(
    name="text_to_speech",
    description="Synthesise speech from text and save it to an audio file.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to speak."},
        },
        "required": ["text"],
    },
)
async def text_to_speech(args: dict, ctx: ToolContext) -> ToolResult:
    if ctx.audio is None or not ctx.audio.available():
        return ToolResult(
            False, "audio is unavailable (install mlx-audio on Apple Silicon)."
        )
    try:
        path = await ctx.audio.speak(args["text"])
    except Exception as exc:
        return ToolResult(False, f"speech synthesis failed: {exc}")
    return ToolResult(True, str(path))
