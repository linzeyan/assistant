#!/usr/bin/env python3
"""Drive the local assistant through one coding brief, unattended.

Streams the SSE turn to a raw log and prints a compact transcript: one line per
tool call, the tail of each failed tool result, and the model's closing text.
The raw log is the record to re-read; stdout is meant to be readable in a
terminal without scrolling past a build.

Exit status: 0 when the turn reached `done`, 1 when it ended on `error`, was
aborted by the probe-loop guard, or the stream stopped early (which is what a
wedged generation looks like from here).

    ./drive.py --brief brief.md --workspace ~/git/project --thinking off --effort low

Only Python's standard library — this has to run against a backend without
being installed alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE = os.environ.get("ASSISTANT_BASE_URL", "http://127.0.0.1:9981")


def short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def visible(text: str) -> str:
    """`text` with the model's scratchpad taken out.

    The backend streams reasoning and answer down one `assistant_delta` channel,
    and some templates (Qwen3.x) end their generation prompt with a bare
    `<think>` — so each pass of the agent loop emits reasoning, a lone
    `</think>`, and then what it meant to say. Concatenated across a dozen
    passes that is a report with the model's deliberations interleaved through
    it, which is exactly what a report read for "did it actually pass" must not
    be.

    Everything before each close is dropped; everything after each is kept. Text
    with no close at all is returned as it stands rather than emptied — a model
    that answered without a scratchpad has said all of it outside one.
    """
    parts = text.split("</think>")
    if len(parts) == 1:
        return text.strip()
    return "\n".join(part.strip() for part in parts[1:] if part.strip())


def get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def resolve_model(base: str, model: str | None) -> str:
    """The model id to drive: the flag, else the backend's configured default.

    `POST /chat` requires a model, and hardcoding one here would pin every user
    of this script to whatever was on the machine it was written on.
    """
    if model:
        return model
    try:
        chosen = (get_json(f"{base}/models/default") or {}).get("default")
    except Exception as exc:
        sys.exit(f"[drive] cannot reach {base} to look up the default model: {exc}")
    if not chosen:
        sys.exit("[drive] no default model is set on the backend — pass --model")
    return chosen


def reconcile_knobs(base: str, model: str, thinking: str | None, effort: str | None):
    """Drop or reject reasoning knobs this model's chat template can't take.

    The template is the authority, and the backend already reads it: GET
    /models/{id}/settings returns `capabilities` when it can tell. An effort
    value the template doesn't know makes some templates raise mid-render, which
    surfaces as a failed turn several seconds in rather than as a bad argument —
    so a wrong value is worth catching here, where the message can name the ones
    that would have worked. `capabilities` is omitted entirely when the backend
    can't tell, and that is not the same as "no knobs": send the flags as given.
    """
    try:
        caps = get_json(
            f"{base}/models/{urllib.parse.quote(model, safe='/')}/settings"
        ).get("capabilities")
    except Exception:
        caps = None
    if caps is None:
        return thinking, effort
    if thinking is not None and not caps.get("thinking"):
        print(f"[drive] {model}'s template has no thinking switch — ignoring --thinking")
        thinking = None
    values = caps.get("effort") or []
    if effort is not None:
        if not values:
            print(f"[drive] {model}'s template never reads reasoning_effort — ignoring --effort")
            effort = None
        elif effort not in values:
            sys.exit(f"[drive] --effort {effort!r} unknown to {model}; it takes: {', '.join(values)}")
    return thinking, effort


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Send one brief to the local assistant and report what it did."
    )
    ap.add_argument("--brief", required=True, help="file holding the message to send")
    ap.add_argument("--workspace", required=True, help="absolute path the tools operate in")
    ap.add_argument("--label", help="names the log files (default: the brief's filename)")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"backend base URL (default {DEFAULT_BASE})")
    ap.add_argument("--model", help="model id; default: the backend's configured default")
    ap.add_argument("--max-iters", type=int, default=60, help="tool-call ceiling for the turn")
    ap.add_argument(
        "--session-file",
        help="read the session id from here if it exists, write it back after; "
        "omit for a fresh conversation",
    )
    ap.add_argument("--out", default="drive-logs", help="directory for the logs (default ./drive-logs)")
    # The two reasoning knobs POST /chat forwards into the chat template. Left
    # unset they inherit whatever the model is configured for, which is what an
    # absent flag should mean — pinning them here would silently override the
    # app's own per-model menu for every run driven from this script.
    ap.add_argument(
        "--thinking",
        choices=("on", "off"),
        help="turn the scratchpad on or off for this turn; omit to inherit",
    )
    ap.add_argument(
        "--effort",
        help="reasoning effort for this turn (values come from the model's template); "
        "omit to inherit",
    )
    ap.add_argument(
        "--probe-limit",
        type=int,
        default=3,
        help="abort after this many consecutive scratch-file bash calls with no edit "
        "between them (0 disables)",
    )
    args = ap.parse_args()

    label = args.label or Path(args.brief).stem
    model = resolve_model(args.base, args.model)
    thinking, effort = reconcile_knobs(args.base, model, args.thinking, args.effort)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / f"{label}.sse.jsonl"
    session_id = None
    if args.session_file:
        Path(args.session_file).parent.mkdir(parents=True, exist_ok=True)
        if Path(args.session_file).exists():
            session_id = Path(args.session_file).read_text().strip() or None

    payload = {
        "message": Path(args.brief).read_text(),
        "model": model,
        "workspace": args.workspace,
        "max_iters": args.max_iters,
    }
    if session_id:
        payload["session_id"] = session_id
    if thinking:
        payload["thinking"] = thinking == "on"
    if effort:
        payload["reasoning_effort"] = effort

    req = urllib.request.Request(
        f"{args.base}/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.time()
    print(
        f"[drive] {label}: model={model} workspace={args.workspace} "
        f"session={session_id or 'new'} thinking={thinking or 'inherit'} "
        f"effort={effort or 'inherit'}"
    )
    raw = raw_path.open("w")
    text_parts: list[str] = []
    status = 1
    calls = 0
    failure: str | None = None
    # Consecutive `bash` calls that write a scratch program, with no edit between
    # them. This is what a diagnosis loop looks like from here: the stream stays
    # healthy and every turn reads like progress, so elapsed time tells you
    # nothing. The tool name is the tell. Observed once at seven turns, by which
    # point the cause had been correctly identified on the first one.
    probes = 0

    def stamp() -> str:
        return f"{time.time() - started:7.1f}s"

    aborted = False
    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            buf = ""
            for chunk in resp:
                if aborted:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    if not line.startswith("data: "):
                        continue
                    raw.write(line[6:] + "\n")
                    raw.flush()
                    ev = json.loads(line[6:])
                    kind = ev.get("type")
                    if kind == "session":
                        session_id = ev["session_id"]
                        if args.session_file:
                            Path(args.session_file).write_text(session_id)
                        print(f"[drive] session {session_id}")
                    elif kind == "assistant_delta":
                        text_parts.append(ev.get("content", ""))
                    elif kind == "tool_call":
                        calls += 1
                        a = ev.get("arguments") or {}
                        res = a.get("command") or a.get("path") or a.get("pattern") or ""
                        print(f"{stamp()}  #{calls:02d} {ev.get('name')}  {short(res, 140)}")
                        sys.stdout.flush()
                        name = ev.get("name", "")
                        if name == "bash" and "/tmp/" in (a.get("command") or ""):
                            probes += 1
                        elif name in ("edit_file", "write_file"):
                            probes = 0
                        if args.probe_limit and probes >= args.probe_limit:
                            failure = (
                                f"{probes} scratch programs in a row and no edit — "
                                "the diagnosis is in the log above; stop and fix it by hand"
                            )
                            print(f"{stamp()}  ABORT: {failure}")
                            status = 1
                            aborted = True
                            break
                    elif kind == "tool_result":
                        if not ev.get("ok"):
                            print(f"{stamp()}      ✗ {ev.get('name')}: "
                                  f"{short(ev.get('content', ''), 400)}")
                            sys.stdout.flush()
                    elif kind == "error":
                        print(f"{stamp()}  ERROR: {ev.get('detail')}")
                        failure = ev.get("detail") or "unspecified error"
                        status = 1
                    elif kind == "done":
                        usage = ev.get("usage", {})
                        print(f"{stamp()}  done  context={usage.get('context_tokens')} "
                              f"output={usage.get('output_tokens')}")
                        status = 0
    except urllib.error.HTTPError as exc:
        failure = f"HTTP {exc.code}: {short(exc.read().decode('utf-8', 'replace'), 300)}"
        print(f"{stamp()}  {failure}")
    except KeyboardInterrupt:
        print("[drive] interrupted")
    finally:
        raw.close()

    answer = visible("".join(text_parts))
    (out / f"{label}.answer.md").write_text(answer)
    (out / f"{label}.raw.md").write_text("".join(text_parts).strip())
    print("\n===== assistant =====")
    print(answer if answer else "(no closing text — a wedged or cancelled turn looks like this)")
    print(f"\n[drive] {calls} tool calls, {time.time() - started:.0f}s, raw: {raw_path}")
    # Last line, and unambiguous. A turn that ends on an error still leaves a
    # closing paragraph above — the model's last words before it was stopped —
    # and read from the bottom of a log that paragraph is indistinguishable from
    # a report of work done. This exists because one such turn was read as a
    # success; the exit status alone was not enough, because a `| tee` in the
    # calling shell takes it.
    print(f"[drive] OUTCOME: {'ok' if status == 0 else 'FAILED — ' + (failure or 'stream ended early')}")
    return status


if __name__ == "__main__":
    sys.exit(main())
