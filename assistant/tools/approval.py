from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Protocol

from .base import Tool


class ApprovalPolicy(Protocol):
    async def approve(self, tool: Tool, arguments: dict) -> bool: ...


# --- wildcard permission rules (spring1 S5) -------------------------------------------

_DECISIONS = frozenset({"allow", "deny", "ask"})
# Argument keys that name the "resource" a tool acts on, most-specific first. Used to match
# a rule's resource glob (a path for file tools, the command for bash, a URL for web).
_RESOURCE_KEYS = ("path", "command", "url", "pattern", "name")


@dataclass(frozen=True)
class Rule:
    """A wildcard permission rule: when ``action`` (glob over the tool name) and ``resource``
    (glob over the tool's resource string) both match, ``decision`` applies — ``allow`` (run
    without prompting), ``deny`` (refuse), or ``ask`` (prompt as usual)."""

    action: str
    resource: str = "*"
    decision: str = "ask"

    @classmethod
    def from_dict(cls, data: dict) -> "Rule":
        decision = str(data.get("decision", "ask")).lower()
        if decision not in _DECISIONS:
            raise ValueError(f"approval rule decision must be allow/deny/ask, got {decision!r}")
        action = data.get("action")
        if not action:
            raise ValueError("approval rule needs an 'action' (tool-name glob)")
        return cls(action=str(action), resource=str(data.get("resource", "*")), decision=decision)

    def matches(self, tool_name: str, resource: str) -> bool:
        return fnmatch(tool_name, self.action) and fnmatch(resource, self.resource)

    def is_blanket_deny(self, tool_name: str) -> bool:
        """True when this rule denies *every* resource for the tool — so the tool can never run
        and can be filtered out of the model's schema entirely (S5), not just refused at call
        time. Resource-specific denies don't qualify: the tool is still usable for others."""
        return (
            self.decision == "deny"
            and self.resource in ("*", "**")
            and fnmatch(tool_name, self.action)
        )


def resource_of(arguments: dict) -> str:
    """Best-effort resource string for a tool call (the thing a rule's resource glob matches):
    the first present of path / command / url / pattern / name."""
    for key in _RESOURCE_KEYS:
        value = arguments.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


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
