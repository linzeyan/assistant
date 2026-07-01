"""Model capability heuristics shared across the app (gateway + HTTP + GUI).

Kept dependency-free (pure string checks, no mlx imports) so the Telegram gateway and the
API layer can both import it without pulling the model runtime — the single source of truth for
"which models are weak at agentic tool calls", surfaced identically in the Telegram /models
picker and the GUI model picker.
"""

from __future__ import annotations

# Reasoning/instruct models that, in practice, don't emit agentic tool calls — they describe
# what they'd do and fabricate success (observed live: DeepSeek-R1-Distill and Qwen3-30B-A3B
# both ran "git diff" with ZERO tool calls; session 780d95b0 fabricated a whole Makefile without
# opening the real one). Flagged in the model pickers so the user doesn't pick one for coding.
# Substring match is deliberately narrow: "qwen3-30b-a3b" matches the thinking variant but NOT
# "qwen3-coder-30b-a3b" (the tool-caller to prefer).
WEAK_TOOL_MARKERS = ("deepseek-r1", "r1-distill", "qwq", "qwen3-30b-a3b", "thinking")


def weak_at_tools(model_id: str) -> bool:
    """True for models that tend to narrate tool use instead of actually calling tools."""
    low = model_id.lower()
    return any(m in low for m in WEAK_TOOL_MARKERS)
