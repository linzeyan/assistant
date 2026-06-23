from __future__ import annotations

import asyncio
import uuid
from typing import Protocol

from .base import Tool


class ApprovalPolicy(Protocol):
    async def approve(self, tool: Tool, arguments: dict) -> bool: ...


class PolicyApprover:
    """Default non-interactive approver.

    Approves tools that don't require approval. For tools that DO (write_file,
    edit_file, bash, and later skill_manage), it approves only when approval is
    globally disabled — i.e. there is no human in the loop. The GUI and Telegram
    gateways will supply interactive approvers that supersede this in later phases.
    """

    def __init__(self, approval_required: bool):
        self._required = approval_required

    async def approve(self, tool: Tool, arguments: dict) -> bool:
        if not tool.needs_approval:
            return True
        return not self._required


class InteractiveApprover:
    """Approver that defers approval-required tools to a human out-of-band.

    The agent loop, seeing ``interactive`` is set, emits an ``approval_request``
    event carrying a token and awaits ``wait(token)``. A separate request (the GUI's
    ``POST /chat/approve``) resolves the matching future via :func:`resolve_pending`.
    Safe tools pass immediately. Mirrors the Telegram inline-button approver but over
    HTTP/SSE. ``pending`` is a registry shared on ``app.state`` so the resolve
    endpoint can find the future created here.
    """

    interactive = True

    def __init__(self, pending: dict[str, asyncio.Future], timeout: float = 300):
        self._pending = pending
        self._timeout = timeout

    def new_request(self) -> str:
        token = uuid.uuid4().hex
        self._pending[token] = asyncio.get_running_loop().create_future()
        return token

    async def wait(self, token: str) -> bool:
        fut = self._pending.get(token)
        if fut is None:
            return False
        try:
            return bool(await asyncio.wait_for(fut, self._timeout))
        except (asyncio.TimeoutError, TimeoutError):
            return False  # no response in time -> deny (fail safe)
        finally:
            self._pending.pop(token, None)

    async def approve(self, tool: Tool, arguments: dict) -> bool:
        return not tool.needs_approval  # only safe tools take this path


def resolve_pending(pending: dict[str, asyncio.Future], token: str, decision: bool) -> bool:
    """Resolve a waiting approval future. Returns whether a pending one was found."""
    fut = pending.get(token)
    if fut is not None and not fut.done():
        fut.set_result(decision)
        return True
    return False
