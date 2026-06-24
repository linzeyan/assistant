from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assistant.images.service import MediaService
    from assistant.memory.provider import MemoryProvider
    from assistant.models.mlx_audio import AudioService
    from assistant.models.mlx_video import VideoService
    from assistant.models.mlx_vlm import VisionService
    from assistant.skills.discovery import SkillStore


@dataclass
class ToolResult:
    ok: bool
    content: str


@dataclass
class ToolContext:
    """Ambient context handed to every tool invocation. ``skills``/``memory`` are
    optional so unit tests (and Phase 1/2 callers) can build a context with just a
    cwd; the skill/memory tools degrade gracefully when they're absent."""

    cwd: Path
    skills: "SkillStore | None" = None
    memory: "MemoryProvider | None" = None
    images: "MediaService | None" = None
    vision: "VisionService | None" = None
    audio: "AudioService | None" = None
    video: "VideoService | None" = None
    # Where over-budget tool output is spilled in full (S4); None = bound in place, no spill.
    output_spill_dir: Path | None = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the function arguments
    handler: Callable[[dict, ToolContext], Awaitable[ToolResult]]
    needs_approval: bool = False
    toolset: str = "core"

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
