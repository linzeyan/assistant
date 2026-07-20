"""KV-cache hit-rate report (spring6 K2): ``python -m assistant.eval.cache_stats``.

Aggregates the per-turn ``generation:`` timing lines the backend already logs (N76)
into the number K2 needs: how much of each prompt was served from the reused KV cache
vs recomputed. A ``cached=0`` turn is either a cold start (new conversation / model
just loaded) or a prefix rebuild (the cacheable prefix drifted — e.g. Claude Code's
system-reminder injection); the log line alone can't tell them apart, so both are
reported and the reader interprets against session context.

Pure parsing/aggregation lives here and is unit-tested; the CLI shell reads the live
backend log (which needs a real deployment to exist, so it isn't covered by CI).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from assistant.config import Settings

# N76's line, e.g.:
#   generation: prompt=2457 (cached=0, prefill=2457) prefill=1.98s decode=1174 tok in 14.05s (83.6 tok/s)
_GENERATION_RE = re.compile(
    r"generation: prompt=(?P<prompt>\d+) \(cached=(?P<cached>\d+), prefill=(?P<prefill>\d+)\) "
    r"prefill=(?P<prefill_s>[\d.]+)s decode=(?P<decode>\d+) tok in (?P<decode_s>[\d.]+)s "
    r"\((?P<tps>[\d.]+) tok/s\)"
)


def parse_generation_lines(text: str) -> list[dict]:
    """Every N76 generation record in ``text``, in log order."""
    return [
        {k: (float(v) if "." in v else int(v)) for k, v in m.groupdict().items()}
        for m in _GENERATION_RE.finditer(text)
    ]


def summarize(records: list[dict]) -> dict:
    """Aggregate cache health over ``records``. Warm = any KV reuse at all."""
    turns = len(records)
    prompt = sum(r["prompt"] for r in records)
    cached = sum(r["cached"] for r in records)
    warm = [r for r in records if r["cached"] > 0]
    cold = turns - len(warm)
    warm_prompt = sum(r["prompt"] for r in warm)
    warm_cached = sum(r["cached"] for r in warm)
    return {
        "turns": turns,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "hit_rate": cached / prompt if prompt else 0.0,
        "warm_turns": len(warm),
        "cold_or_rebuilt_turns": cold,
        "warm_hit_rate": warm_cached / warm_prompt if warm_prompt else 0.0,
        "prefill_seconds": sum(r["prefill_s"] for r in records),
        "decode_tokens": sum(r["decode"] for r in records),
    }


def format_report(s: dict) -> str:
    lines = [
        "kv-cache hit rate (from backend generation log)",
        "─" * 47,
        f"turns: {s['turns']}  (warm: {s['warm_turns']}, cold-or-rebuilt: "
        f"{s['cold_or_rebuilt_turns']})",
        f"prompt tokens: {s['prompt_tokens']}  cached: {s['cached_tokens']}  "
        f"hit rate: {s['hit_rate']:.1%}",
        f"warm-turn hit rate: {s['warm_hit_rate']:.1%}",
        f"prefill time paid: {s['prefill_seconds']:.1f}s  decode tokens: "
        f"{s['decode_tokens']}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    default_log = Settings().log_dir / "backend.log"
    p = argparse.ArgumentParser(
        prog="python -m assistant.eval.cache_stats",
        description="Aggregate KV-cache hit rate from the backend's generation log (spring6 K2).",
    )
    p.add_argument("--log", default=str(default_log), help=f"log file (default {default_log})")
    args = p.parse_args(argv)
    path = Path(args.log)
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return 2
    records = parse_generation_lines(path.read_text(encoding="utf-8", errors="replace"))
    if not records:
        print(
            "no generation lines found — the backend logs them only on the mlx-lm path "
            "(VLM-loaded models have no cache, N75), and only since N76.",
            file=sys.stderr,
        )
        return 1
    print(format_report(summarize(records)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
