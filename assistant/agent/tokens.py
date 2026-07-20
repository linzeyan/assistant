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

# English averages ~4 chars/token; non-ASCII (CJK especially) tokenizes near 1 token/char,
# so it is counted per character below. Weighting it at chars/4 under-budgeted dense scripts
# ~4x — enough for one fetched CJK page to slip a ~20k-token bomb past every budget that
# trusted this number (N94). Over-counting accented Latin a little is fine: every caller uses
# the estimate to decide when to trim, where high is safe and low is not.
_CHARS_PER_TOKEN = 4
# Each message carries role markers / chat-template delimiters the content count misses.
_PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str | None) -> int:
    """Conservative token estimate: ASCII at ~4 chars/token, everything else at 1 token/char."""
    if not text:
        return 0
    if text.isascii():
        return -(-len(text) // _CHARS_PER_TOKEN)  # ceil division
    ascii_chars = sum(ch.isascii() for ch in text)
    return -(-ascii_chars // _CHARS_PER_TOKEN) + (len(text) - ascii_chars)


def cut_at_tokens(text: str, max_tokens: int) -> str:
    """Longest prefix of ``text`` that fits ``max_tokens`` under the same ASCII/non-ASCII
    weighting as :func:`estimate_tokens` (walked in quarter-token units so the accounting
    stays integer-only). Returns ``text`` unchanged when it already fits — callers append
    their own truncation marker only on an actual cut."""
    if estimate_tokens(text) <= max_tokens:
        return text
    budget = max_tokens * _CHARS_PER_TOKEN
    used = 0
    for i, ch in enumerate(text):
        used += 1 if ch.isascii() else _CHARS_PER_TOKEN
        if used > budget:
            return text[:i]
    return text


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
