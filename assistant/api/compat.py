"""Shared translation for the OpenAI- and Anthropic-compatible API shims.

These endpoints expose the local model service as a *raw* chat-completions backend so external
agents (Claude Code via ANTHROPIC_BASE_URL, OpenAI-style clients) can drive the local MLX models
directly. They deliberately bypass the assistant's own AgentLoop/tools/skills/memory — the external
client brings its own agent loop, so layering ours underneath would mean two agents fighting. All
that's needed is: messages+tools in, model tokens (and tool calls) out.

The model service already speaks an OpenAI-ish shape (``messages`` list, ``tools`` with a
``function`` envelope, ``{"type": "text"|"tool_calls"}`` events), so the OpenAI path is nearly a
passthrough; the Anthropic path needs real translation (system as a top-level field, content
blocks, tool_use/tool_result), kept here as pure functions so it can be unit-tested.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger("assistant")

_CHATTABLE = ("llm", "vlm")

# Claude Code's thinking budgets, bucketed: "think" ≈ 4k tokens, "think hard" ≈ 10k,
# "ultrathink" ≈ 32k. A local model has no token budget to spend — its chat template exposes an
# effort ladder instead — so the budget is read as the *degree* the user asked for and mapped onto
# whatever rungs that particular model publishes.
_BUDGET_BUCKETS = (8_000, 24_000)


async def resolve_model(service, requested: str) -> str:
    """Map a client-sent model name to a local chat-model id.

    Clients send short ids — Claude Code's ``ANTHROPIC_DEFAULT_*_MODEL`` is e.g.
    ``Qwen3-Coder-30B-A3B-Instruct-8bit`` with no ``mlx-community/`` prefix. Match by exact id,
    then by basename (suffix after ``/``), then by substring, against chat-capable models. Fall
    back to the requested string unchanged so the model service raises its own clear error rather
    than us masking it.
    """
    try:
        ids = [m.id for m in await service.list_models() if m.type in _CHATTABLE]
    except Exception:
        return requested
    if requested in ids:
        return requested
    for mid in ids:
        if mid.split("/")[-1] == requested:
            return mid
    for mid in ids:
        if requested and requested in mid:
            return mid
    return requested


def _text_of(content) -> str:
    """Flatten an Anthropic message ``content`` (str, or a list of blocks) to plain text,
    keeping only the human-readable parts (text blocks, and tool_result payloads)."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "tool_result":
            parts.append(_text_of(block.get("content", "")))
    return "".join(parts)


def anthropic_tools_to_openai(tools) -> list[dict] | None:
    """Anthropic tool schema → the OpenAI ``function`` envelope the model service expects.
    Anthropic: ``{name, description, input_schema}``; OpenAI: ``{type, function:{name,
    description, parameters}}``."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict) or not t.get("name"):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return out or None


def anthropic_to_openai_messages(system, messages: list[dict]) -> list[dict]:
    """Translate an Anthropic ``/v1/messages`` request (top-level ``system`` + ``messages`` whose
    content may be blocks) into the flat OpenAI message list the model service consumes.

    - ``system`` (str or list of text blocks) → a leading ``{"role": "system"}`` message.
    - assistant ``tool_use`` blocks → an assistant message carrying OpenAI ``tool_calls``.
    - user ``tool_result`` blocks → ``{"role": "tool", ...}`` messages so the model sees the
      tool output (the linkage id is preserved for the client; the model just needs the content).
    - everything else collapses to text.
    """
    out: list[dict] = []
    sys_text = _text_of(system) if system else ""
    if sys_text:
        out.append({"role": "system", "content": sys_text})
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            out.append({"role": role, "content": ""})
            continue
        tool_calls = []
        tool_results = []  # emitted as separate role:tool messages (OpenAI shape)
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _text_of(block.get("content", "")),
                })
            # image blocks are dropped: the raw passthrough targets text models.
        if tool_calls:
            out.append({
                "role": "assistant",
                "content": "".join(text_parts),
                "tool_calls": tool_calls,
            })
            out.extend(tool_results)
        else:
            # Tool outputs answer the PRIOR assistant tool_calls, so they must precede any new user
            # text in this same turn — the OpenAI/chat-template ordering. Emitting the user's text
            # before its role:tool answers breaks strict templates (the N72 failure class).
            out.extend(tool_results)
            if text_parts:
                out.append({"role": role, "content": "".join(text_parts)})
    return out


def sampling_params(body: dict) -> dict:
    """Pull the generation knobs both APIs share into model-service ``**params`` (omitting None so
    per-model/global defaults still apply)."""
    params: dict = {}
    if body.get("max_tokens") is not None:
        params["max_tokens"] = body["max_tokens"]
    for k in ("temperature", "top_p", "top_k"):
        if body.get(k) is not None:
            params[k] = body[k]
    return params


async def _accepted_efforts(service, model: str) -> list[str] | None:
    """The effort values this model's chat template accepts; ``None`` when the backend can't say.

    The distinction matters: ``[]`` means the template never reads ``reasoning_effort`` (drop the
    key — at best inert, and it can't be validated), while ``None`` means we simply don't know, so
    an explicit client value is forwarded on the client's own authority.
    """
    probe = getattr(service, "template_capabilities", None)
    if probe is None:
        return None
    try:
        return list((await probe(model)).get("effort") or [])
    except Exception:
        return None


def _bucket_budget(budget: int, values: list[str]) -> str:
    """A thinking budget → one rung of this model's own effort ladder (weakest/middle/strongest)."""
    rank = sum(budget >= b for b in _BUDGET_BUCKETS)
    return [values[0], values[len(values) // 2], values[-1]][rank]


async def reasoning_overrides(service, model: str, body: dict) -> dict:
    """Per-request chat-template overrides from either API's reasoning knobs.

    This is what makes "think harder" mean something locally: Claude Code's ``thinking`` block and
    an OpenAI client's ``reasoning_effort`` both land on the same two template variables the GUI's
    per-conversation menu drives, and beat the model's saved defaults for that one request.

    An effort value the model would reject is dropped with a warning rather than forwarded: some
    templates ``raise_exception`` on an unknown value, which would kill the turn mid-render — a
    worse answer to a client's bad guess than quietly using the model's own default.
    """
    out: dict = {}
    budget: int | None = None
    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        # Anthropic sends the block only to state an intent, so "disabled" is as explicit as
        # "enabled" — a client that asked for no thinking must not get a thinking model anyway.
        if thinking.get("type") in ("enabled", "disabled"):
            out["enable_thinking"] = thinking["type"] == "enabled"
        b = thinking.get("budget_tokens")
        if out.get("enable_thinking") and isinstance(b, int) and not isinstance(b, bool):
            budget = b

    effort = body.get("reasoning_effort")
    if not isinstance(effort, str) or not effort:
        effort = None
    if effort is None and budget is None:
        return out

    values = await _accepted_efforts(service, model)
    if effort is not None:  # an explicit value outranks one inferred from a budget
        if values is None or effort in values:
            out["reasoning_effort"] = effort
        else:
            log.warning(
                "ignoring reasoning_effort=%r for %s: template accepts %s",
                effort, model, values or "no effort values",
            )
    elif values:
        out["reasoning_effort"] = _bucket_budget(budget, values)
    return out
