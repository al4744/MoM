"""Generate synthetic MoM trace JSON files.

The emitted schema matches ``evaluation.trace_loader.TraceSpec`` and can be
consumed by custom harnesses or inspected directly. ``evaluation/run_eval.py``
still builds in-memory fixture traces from YAML by default, so this generator is
an additive Workstream C artifact rather than a behavioral dependency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


_LATENCY_MS = {
    "fast": (25.0, 40.0),
    "medium": (250.0, 400.0),
    "long": (1500.0, 2500.0),
}
_MIXED_TOOLS = ("search", "calculator", "database")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic trace JSON.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, nargs="+", default=[5, 10, 25, 50])
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192, 16384],
        help="Prompt-token budgets to sweep.",
    )
    parser.add_argument(
        "--tool-latency-dist",
        choices=("fast", "medium", "long", "mixed"),
        default="mixed",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B",
    )
    parser.add_argument("--tool-every", type=int, default=3)
    parser.add_argument("--expected-output-tokens", type=int, default=64)
    return parser.parse_args()


def build_trace(
    *,
    trace_id: str,
    model: str,
    turns: int,
    context_length: int,
    tool_latency_dist: str,
    tool_every: int,
    expected_output_tokens: int,
) -> dict:
    trace_turns: list[dict] = []
    prompt_tokens_per_turn = max(context_length // max(turns, 1), 1)
    idx = 0
    while idx < turns:
        trace_turns.append(
            {
                "turn_index": idx,
                "kind": "user_prompt",
                "tokens": prompt_tokens_per_turn,
                "tool_name": None,
                "tool_latency_ms": None,
                "expected_output_tokens": expected_output_tokens,
            }
        )
        idx += 1
        if idx < turns and tool_every > 0 and idx % tool_every == 0:
            tool_name = _MIXED_TOOLS[(idx // tool_every - 1) % len(_MIXED_TOOLS)]
            trace_turns.append(
                {
                    "turn_index": idx,
                    "kind": "tool_call",
                    "tokens": 32,
                    "tool_name": tool_name,
                    "tool_latency_ms": _tool_latency_ms(tool_latency_dist, idx),
                    "expected_output_tokens": 0,
                }
            )
            idx += 1
            if idx < turns:
                trace_turns.append(
                    {
                        "turn_index": idx,
                        "kind": "tool_return",
                        "tokens": 128,
                        "tool_name": tool_name,
                        "tool_latency_ms": None,
                        "expected_output_tokens": 0,
                    }
                )
                idx += 1

    return {
        "trace_id": trace_id,
        "model": model,
        "prompt_tokens": context_length,
        "tool_latency_dist": tool_latency_dist,
        "turns": trace_turns[:turns],
    }


def write_traces(args: argparse.Namespace) -> list[Path]:
    args.output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for turns in args.turns:
        for context_length in args.context_lengths:
            trace_id = f"{context_length // 1024}k-{turns}turn-{args.tool_latency_dist}"
            trace = build_trace(
                trace_id=trace_id,
                model=args.model,
                turns=turns,
                context_length=context_length,
                tool_latency_dist=args.tool_latency_dist,
                tool_every=args.tool_every,
                expected_output_tokens=args.expected_output_tokens,
            )
            path = args.output / f"{trace_id}.json"
            path.write_text(json.dumps(trace, indent=2) + "\n")
            paths.append(path)
    return paths


def _tool_latency_ms(dist: str, turn_index: int) -> float:
    if dist == "mixed":
        dist = ("fast", "medium", "long")[turn_index % 3]
    low, high = _LATENCY_MS[dist]
    # Deterministic midpoint/jitter pattern; no random seed needed.
    frac = ((turn_index * 17) % 100) / 100.0
    return low + (high - low) * frac


def main() -> int:
    paths = write_traces(parse_args())
    for path in paths:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
