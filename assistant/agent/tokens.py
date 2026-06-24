"""Token estimation (spring1 prerequisite for compaction).

There is no exact tokenizer at this layer — the loaded MLX model has one, but reaching it
from the agent loop would couple the loop to a specific backend for what is only a budgeting
decision. So this is a deliberately conservative ``chars / 4`` heuristic plus a small
per-message overhead for role tags and chat-template delimiters. It is used to decide WHEN
to compact and to surface a context-usage number — both tolerate approximation; neither needs
exactness. A real per-model tokenizer can replace ``estimate_tokens`` later without changing
callers.
"""

from __future__ import annotations

import json

# English averages ~4 chars/token; CJK is denser (~1.5), so 4 is conservative-ish for mixed
# text. Compaction keeps a reserve margin on top, so a rough estimate is safe.
_CHARS_PER_TOKEN = 4
# Each message carries role markers / chat-template delimiters the content count misses.
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str | None) -> int:
    """Conservative token estimate for a single string (ceil of chars / 4)."""
    if not text:
        return 0
    return -(-len(text) // _CHARS_PER_TOKEN)  # ceil division


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate the token footprint of an OpenAI-style message list — the number compaction
    compares against the context window. Counts string content, plus any assistant
    ``tool_calls`` (their serialized function name + arguments), plus per-message overhead."""
    total = 0
    for m in messages:
        total += _PER_MESSAGE_OVERHEAD
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        for tc in m.get("tool_calls") or []:
            total += estimate_tokens(json.dumps(tc.get("function", {}), ensure_ascii=False))
    return total
