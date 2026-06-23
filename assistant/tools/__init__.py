from .registry import ToolRegistry, registry


def build_registry() -> ToolRegistry:
    """Return the populated tool registry.

    Imports the builtin tool modules for their decorator side effects. This is an
    EXPLICIT call rather than an import-time scan, to avoid hermes-agent's pitfall
    where plugin discovery only ran as a side effect of importing an unrelated module.
    """
    from . import (  # noqa: F401  (import = self-register)
        audio_tools,
        coding,
        image_tool,
        memory_tools,
        shell,
        skill_tools,
        video_tool,
        vision_tool,
        web_tools,
    )

    return registry
