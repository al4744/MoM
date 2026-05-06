"""Workstream D — CLI wrapper for the concurrent runner.

Drives N concurrent traces through one vLLM engine and writes the same
output shape as evaluation/run_eval.py so comparison_table.py and the
existing make targets work unchanged.

Usage:
    PYTHONPATH=. python3 scripts/run_concurrent_eval.py \\
        --config configs/retention_constrained.yaml \\
        --output results/concurrent-$TS/retention/ \\
        --concurrency 4

Replication strategy:
    Each YAML trace spec is replicated `--num-traces` times (default:
    `--concurrency`). With concurrency=4 and 4 trace shapes (5/10/25/50
    turn) in the YAML, we get 16 trace runs total, with up to 4 in flight
    at once.

The output of this script is a results directory with per-trace JSONs +
a summary.json, identical in shape to a regular run_eval invocation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a script: scripts/run_concurrent_eval.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.concurrent_runner import run_concurrent
from evaluation.engine_adapter import build_real_engine
from evaluation.metrics import RunSummary
from evaluation.run_eval import _configure_retention_events, load_config
from evaluation.trace_loader import fixture_trace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a concurrent multi-trace benchmark.")
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="MoM YAML config (configs/*.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for per-trace JSON + summary.json.",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max number of in-flight requests at any time.",
    )
    p.add_argument(
        "--num-traces",
        type=int,
        default=None,
        help="How many replicas of each YAML trace to run. "
             "Defaults to --concurrency.",
    )
    p.add_argument(
        "--mock-engine",
        action="store_true",
        help="Use evaluation.engine_adapter.MockEngine (no CUDA, for smoke testing).",
    )
    return p.parse_args()


def build_specs(cfg: dict[str, Any], num_traces: int) -> list:
    """Replicate each YAML trace spec into ``num_traces`` independent agents."""
    model = cfg.get("model", {}).get("name", "unknown-model")
    trace_descs = cfg.get("traces", [])
    specs = []
    for desc in trace_descs:
        for replica in range(num_traces):
            specs.append(
                fixture_trace(
                    trace_id=f"{desc['id']}-r{replica}",
                    num_turns=desc.get("turns", 10),
                    model=model,
                    prompt_tokens=desc.get("prompt_tokens", 1024),
                    tool_latency_ms=desc.get("tool_latency_ms", 0.0),
                )
            )
    return specs


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    _configure_retention_events(args.output, use_wandb=False)

    num_traces = args.num_traces or args.concurrency
    specs = build_specs(cfg, num_traces=num_traces)

    config_name = cfg.get("name", args.config.stem)
    print(f"=== Concurrent run ===")
    print(f"  config       : {config_name}")
    print(f"  total specs  : {len(specs)} ({num_traces} replicas × {len(cfg.get('traces', []))} shapes)")
    print(f"  concurrency  : {args.concurrency}")
    print(f"  output       : {args.output}")
    print()

    if args.mock_engine:
        from evaluation.engine_adapter import MockEngine
        engine = MockEngine()
    else:
        engine = build_real_engine(cfg)

    traces = run_concurrent(
        engine=engine,
        specs=specs,
        cfg=cfg,
        concurrency=args.concurrency,
    )

    # Per-trace JSON dump.
    for tr in traces:
        out_path = args.output / f"{tr.trace_id}.json"
        tr.emit_json(out_path)
    print(f"  wrote {len(traces)} per-trace JSON files")

    # Aggregate.
    summary = RunSummary.from_traces(traces)
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.__dict__, f, indent=2)
    print(f"  wrote {summary_path}")

    print()
    print(
        f"  config={config_name} concurrency={args.concurrency} "
        f"traces={summary.num_traces} ttft_ms={summary.mean_ttft_ms:.2f} "
        f"prt_ms={summary.mean_prefill_recomp_ms:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
