"""Live tool-calling reliability harness (spring5 A1): ``python -m assistant.eval.measure``.

Drives a *running* backend over a small prompt set N times, reads each turn's trace, and prints a
success rate with failures bucketed into the four death modes (see ``reliability``). This is the
step that produces real numbers on the very models the user runs — it MUST hit a live backend
(``make app-run`` / the packaged app), because the whole point is to measure real MLX models, not a
fake. Optionally appends each raw model output to a corpus JSONL so real failures sediment into the
regression set (``tests/test_tool_parsing_corpus.py``).

Examples:
    python -m assistant.eval.measure --model Qwen3-Coder-30B -n 10
    python -m assistant.eval.measure --prompts my_prompts.txt --append-corpus captures.jsonl

The pure classification/aggregation lives in ``reliability`` and is unit-tested; everything here is
the I/O shell (HTTP + argparse), which needs the live backend and so isn't covered by CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from assistant.config import Settings
from assistant.eval import reliability


async def _run_one(client: httpx.AsyncClient, model: str, message: str, timeout: float) -> str | None:
    """POST one chat turn, drain its SSE stream, and return the resulting session id (or None on a
    transport failure). The turn's trace is fetched separately so classification reuses the loop's
    own outcome rather than re-deriving it from the event stream."""
    session_id: str | None = None
    body = {"message": message, "model": model}
    async with client.stream("POST", "/chat", json=body, timeout=timeout) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            if event.get("type") == "session":
                session_id = event["session_id"]
            # Drain to completion; terminal events (done/error) just end the stream.
    return session_id


async def _latest_trace(client: httpx.AsyncClient, session_id: str) -> dict | None:
    """Full trace for the session's most recent turn (the one we just drove), or None."""
    turns = (await client.get(f"/sessions/{session_id}/turns")).json().get("turns") or []
    if not turns:
        return None  # tracing disabled, or the turn wasn't recorded
    turn_id = turns[0]["turn_id"]  # list_for_session is newest-first
    resp = await client.get(f"/turns/{turn_id}")
    return resp.json() if resp.status_code == 200 else None


async def _measure(args: argparse.Namespace, prompts: list[str]) -> int:
    base_url = args.base_url
    buckets: list[str] = []
    captures: list[dict] = []
    async with httpx.AsyncClient(base_url=base_url) as client:
        model = args.model or await _autodetect_model(client)
        if not model:
            print("no model given and none is loaded; pass --model", file=sys.stderr)
            return 2
        print(f"measuring {model} over {len(prompts)} prompts × {args.runs} runs at {base_url}\n",
              file=sys.stderr)
        for prompt in prompts:
            for _ in range(args.runs):
                try:
                    session_id = await _run_one(client, model, prompt, args.timeout)
                except (httpx.HTTPError, json.JSONDecodeError) as exc:
                    print(f"  request failed ({exc}); counting as crash", file=sys.stderr)
                    buckets.append(reliability.CRASH)
                    continue
                trace = await _latest_trace(client, session_id) if session_id else None
                if trace is None:
                    print("  no trace (is trace_enabled set?); skipping", file=sys.stderr)
                    continue
                buckets.append(reliability.classify_turn(trace, expects_tool=True))
                captures.append(_capture_row(trace))

    if not buckets:
        print("no turns measured — is the backend running with tracing enabled?", file=sys.stderr)
        return 1
    summary = reliability.summarize(buckets)
    print(reliability.format_report(summary, model=args.model, runs=args.runs))
    if args.append_corpus:
        _append_corpus(Path(args.append_corpus), captures)
        print(f"\nappended {len(captures)} captures to {args.append_corpus}", file=sys.stderr)
    return 0


async def _autodetect_model(client: httpx.AsyncClient) -> str | None:
    """Pick the currently-loaded chat model, else the first listed one — a convenience so the
    harness can run without --model against whatever the backend already has hot."""
    models = (await client.get("/models")).json().get("models") or []
    loaded = next((m["id"] for m in models if m.get("loaded")), None)
    return loaded or (models[0]["id"] if models else None)


def _capture_row(trace: dict) -> dict:
    """A corpus-shaped row: the raw model text per step, the parsed calls (to replay against
    ``parse_tool_calls`` for a parse_miss), and the tool results (to see *which* tool failed on a
    tool_error — mode 3 — without re-running)."""
    return {
        "model": trace.get("model"),
        "user_text": trace.get("user_text"),
        "outcome": trace.get("outcome"),
        "steps": [
            {
                "model_text": s.get("model_text", ""),
                "parsed_calls": s.get("parsed_calls", []),
                "tool_results": [
                    {"name": r.get("name"), "ok": r.get("ok"),
                     "content": (r.get("content") or "")[:200]}
                    for r in s.get("tool_results") or []
                ],
            }
            for s in trace.get("steps") or []
        ],
    }


def _append_corpus(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_prompts(spec: str | None) -> list[str]:
    if not spec:
        return list(reliability.DEFAULT_PROMPTS)
    # One prompt per non-blank line; '#' lines are comments.
    lines = Path(spec).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def main(argv: list[str] | None = None) -> int:
    settings = Settings()
    default_base = f"http://{settings.backend_host}:{settings.backend_port}"
    p = argparse.ArgumentParser(
        prog="python -m assistant.eval.measure",
        description="Measure tool-calling reliability against a running backend (spring5 A1).",
    )
    p.add_argument("--base-url", default=default_base, help=f"backend URL (default {default_base})")
    p.add_argument("--model", default=None, help="model id (default: the loaded one, via /models)")
    p.add_argument("-n", "--runs", type=int, default=10, help="runs per prompt (default 10)")
    p.add_argument("--prompts", default=None, help="file with one prompt per line (default: built-in set)")
    p.add_argument("--append-corpus", default=None, help="append raw captures to this JSONL")
    p.add_argument("--timeout", type=float, default=600.0, help="per-turn HTTP timeout seconds")
    args = p.parse_args(argv)
    return asyncio.run(_measure(args, _load_prompts(args.prompts)))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
