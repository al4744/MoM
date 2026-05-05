"""Workstream D — Evaluation runner CLI.

Dispatches a benchmark run for a single (config, trace_set) pair, collects
TraceResults, writes a per-trace JSON dump, and emits a markdown summary.

Usage:
    python evaluation/run_eval.py \\
        --config configs/baseline.yaml \\
        --output results/baseline-2026-05-05/

This module is intentionally a thin orchestration layer. The actual engine
hookups live in workstream-owned modules:

  - configs/*.yaml             → parsed here, dispatched to engine factory
  - benchmarks/trace_generator → produces synthetic traces (Workstream C)
  - src/retention/events.py    → per-turn instrumentation (Workstream A)
  - evaluation/metrics.py      → metric dataclasses (this module)
  - evaluation/comparison_table → markdown emitter (post-run)

Until A/B/C land their engine hooks, this script runs in --dry-run mode and
emits stub TraceResults so the eval matrix structure is testable end-to-end.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evaluation.metrics import RunSummary, TraceResult, TurnMetrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the MoM evaluation suite for one configuration.",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file under configs/.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for per-trace JSON + summary markdown.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip engine invocation; emit stub traces. Used until A/B/C land hooks.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Push aggregated RunSummary to WandB (requires WANDB_API_KEY).",
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config. Soft-imports yaml so dry-run works without it."""
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            f"PyYAML required to load {path}; pip install pyyaml"
        ) from e
    with open(path) as f:
        return yaml.safe_load(f)


def stub_trace(config_name: str, trace_id: str, num_turns: int) -> TraceResult:
    """Generate a deterministic stub TraceResult.

    Used only in --dry-run mode so the eval pipeline can be exercised before
    Workstreams A/B/C land their engine hooks. Numbers are sentinels (negative
    values intentional) so they are obvious in any downstream comparison.
    """
    trace = TraceResult(
        config_name=config_name,
        trace_id=trace_id,
        peak_vram_mb=-1.0,
        preemption_count=-1,
        metadata={"stub": True},
    )
    for i in range(num_turns):
        trace.turns.append(
            TurnMetrics(
                turn_index=i,
                ttft_ms=-1.0,
                tbt_ms_mean=-1.0,
                tbt_ms_p99=-1.0,
                wallclock_ms=-1.0,
                prefill_recomp_ms=(-1.0 if i > 0 and i % 3 == 0 else None),
                tool_name=("search" if i % 3 == 0 else None),
            )
        )
    return trace


def run_real(cfg: dict[str, Any]) -> list[TraceResult]:
    """Actual engine invocation. Stubbed until A/B/C land their hooks.

    When implemented, this should:
      1. Build the trace set via benchmarks.trace_generator (Workstream C).
      2. Construct the vLLM engine with cfg['engine'] (Workstream A/B/C wiring).
      3. For each trace: run engine, collect events from src/retention/events,
         build TraceResult.
    """
    raise NotImplementedError(
        "Real engine hooks not yet available. "
        "Run with --dry-run until Workstreams A/B/C land integration."
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    config_name = cfg.get("name", args.config.stem)
    trace_specs = cfg.get("traces", [])
    if not trace_specs:
        print(f"[!] No traces defined in {args.config}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        traces = [
            stub_trace(config_name, ts["id"], ts.get("turns", 10))
            for ts in trace_specs
        ]
    else:
        traces = run_real(cfg)

    # Per-trace JSON dump.
    for tr in traces:
        out_path = args.output / f"{tr.trace_id}.json"
        tr.emit_json(out_path)
        print(f"  wrote {out_path}")

    # Aggregate summary.
    summary = RunSummary.from_traces(traces)
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.__dict__, f, indent=2)
    print(f"  wrote {summary_path}")

    # Optional WandB push.
    if args.wandb:
        try:
            import wandb  # type: ignore
            wandb.init(project="mom-eval", name=config_name, config=cfg)
            wandb.log(summary.__dict__)
            wandb.finish()
        except ImportError:
            print("[!] wandb not installed; skipping --wandb push", file=sys.stderr)

    print(f"\n  config={config_name} traces={len(traces)} ttft_ms={summary.mean_ttft_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
