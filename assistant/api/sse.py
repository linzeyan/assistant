"""SSE keepalive wrapper shared by the streaming chat routes.

A turn has long, legitimately-silent phases — queueing behind the single engine slot,
prefilling a long prompt, and tool-call markup being buffered by the marker suppressor
(a slow model's whole first response can be one suppressed tool call). During those
phases zero bytes reach the client, so any sane client/proxy read-timeout kills a
healthy turn — and the abandoned generation then burns the engine slot (N81). A
periodic heartbeat keeps the connection observably alive without touching the event
protocol: ``/chat`` sends an SSE comment (every parser ignores lines starting with
``:``), ``/v1/messages`` sends Anthropic's documented ``ping`` event.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

COMMENT_KEEPALIVE = ": keepalive\n\n"
KEEPALIVE_INTERVAL = 15.0


async def with_keepalive(
    events: AsyncIterator[str],
    payload: str = COMMENT_KEEPALIVE,
    interval: float = KEEPALIVE_INTERVAL,
) -> AsyncIterator[str]:
    """Forward ``events``, injecting ``payload`` whenever ``interval`` seconds pass
    without one. Transport-level only: event content and ordering are untouched."""
    it = events.__aiter__()
    pending: asyncio.Task | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(it))
            try:
                # shield: a timeout must not cancel the in-flight anext — the next
                # wait resumes it. Client disconnect cancels US, handled in finally.
                item = await asyncio.wait_for(asyncio.shield(pending), interval)
            except TimeoutError:
                yield payload
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            yield item
    finally:
        # Client gone (or consumer closed us): the inner generator's own cleanup
        # (checkpoint logging, engine stop) must still run. Mid-anext, cancelling the
        # task delivers CancelledError inside it; idle at a yield, close it directly.
        if pending is not None and not pending.done():
            pending.cancel()
        elif (aclose := getattr(it, "aclose", None)) is not None:
            await aclose()
