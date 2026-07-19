from __future__ import annotations

import asyncio

import pytest

from assistant.api.sse import with_keepalive


async def test_fast_events_pass_through_without_keepalive():
    async def gen():
        yield "a"
        yield "b"

    out = [e async for e in with_keepalive(gen(), payload="PING", interval=5.0)]
    assert out == ["a", "b"]


async def test_silent_gap_injects_keepalive():
    async def gen():
        yield "a"
        await asyncio.sleep(0.08)
        yield "b"

    out = [e async for e in with_keepalive(gen(), payload="PING", interval=0.02)]
    assert out[0] == "a" and out[-1] == "b"
    assert "PING" in out[1:-1]  # at least one heartbeat during the gap
    assert [e for e in out if e != "PING"] == ["a", "b"]  # events untouched, in order


async def test_disconnect_mid_wait_cancels_inner_generator():
    # The real disconnect shape: the consumer task is cancelled while awaiting the next
    # event. Cancellation must reach the inner generator (routes_chat logs and re-raises
    # on exactly this), or its cleanup — checkpoint logging, engine stop — never runs.
    state = {"cancelled": False}

    async def gen():
        try:
            yield "a"
            await asyncio.sleep(30)
            yield "never"
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    async def consume():
        async for _ in with_keepalive(gen(), payload="PING", interval=0.01):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.05)  # consume "a", then block inside the inner sleep
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.02)  # let the cancelled anext task unwind
    assert state["cancelled"]


async def test_close_while_idle_still_closes_inner_generator():
    # Wrapper closed while suspended at a yield (no anext in flight): the inner
    # generator must be aclosed so its finally runs, not orphaned to the GC.
    state = {"closed": False}

    async def gen():
        try:
            yield "a"
            yield "b"
        finally:
            state["closed"] = True

    wrapped = with_keepalive(gen(), payload="PING", interval=5.0)
    assert await wrapped.__anext__() == "a"
    await wrapped.aclose()
    assert state["closed"]
