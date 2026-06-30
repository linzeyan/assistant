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
    # Sync, thread-safe progress sink set by the agent loop for each tool run: a long tool
    # (e.g. video denoising in a worker thread) calls on_progress(fraction, label) and the
    # loop surfaces each tick as a tool_progress event. None when nothing consumes progress.
    on_progress: "Callable[[float, str], None] | None" = None
    # Where over-budget tool output is spilled in full (S4); None = bound in place, no spill.
    output_spill_dir: Path | None = None


def service_available(attr: str) -> "Callable[[ToolContext], bool]":
    """Build a schema-gate predicate for a tool that needs the ``ctx.<attr>`` media service.

    The tool is offered to the model only when that service exists and reports ``available()`` —
    a capability check (the dep is importable), so an unloaded/uninstalled vision/audio/video
    backend never appears in the schema, can't be called, and so can't fail at runtime (death
    mode 3). It does NOT hide a tool whose model merely isn't loaded yet: ``available()`` is True
    as long as the backend can load one on demand."""

    def check(ctx: "ToolContext") -> bool:
        svc = getattr(ctx, attr, None)
        return svc is not None and svc.available()

    return check


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema for the function arguments
    handler: Callable[[dict, ToolContext], Awaitable[ToolResult]]
    needs_approval: bool = False
    toolset: str = "core"
    # Optional schema-time availability gate (G/S13): when set and it returns False for the turn's
    # context, the tool is omitted from the schema sent to the model. None = always offered.
    check_fn: "Callable[[ToolContext], bool] | None" = None

    def to_openai(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
