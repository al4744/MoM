"""Workstream D — CLI wrapper for the concurrent runner.

Drives N concurrent traces through one vLLM engine and writes the same
output shape as evaluation/run_eval.py so comparison_table.py and the
existing make targets work unchanged.

Modes (selected by --workload-class):

  legacy           Replicate cfg.traces ``--num-traces`` times. Default if
                   no other workload flag is given.
  filler_focal     1 measured focal trace + N continuous filler traces.
                   Daksh's microbenchmark — the regime retention was
                   designed for. Trigger via either ``--workload-class
                   filler_focal --num-fillers N`` or the legacy shortcut
                   ``--num-fillers N``.
  lockstep         N identical agents, all start t=0. Negative control —
                   no cross-agent contention pattern retention can exploit.
  staggered        N agents arrive Poisson(rate). Sustained-load realistic
                   regime.
  heterogeneous    N agents start together, log-normal tool latencies.
                   Exercises the per-tool-EMA TTL predictor.
  burst            N agents start within a tight burst_duration_ms window.
                   Transient burst — cache fills sharply.

Usage examples:

    # Filler+focal (existing):
    PYTHONPATH=. python3 scripts/run_concurrent_eval.py \\
        --config configs/retention_filler.yaml \\
        --output results/filler-$TS/retention/ \\
        --workload-class filler_focal \\
        --num-fillers 7 --focal-num-turns 4 --focal-tool-latency-ms 5000 \\
        --concurrency 8

    # Staggered:
    PYTHONPATH=. python3 scripts/run_concurrent_eval.py \\
        --config configs/retention_filler.yaml \\
        --output results/staggered-$TS/retention/ \\
        --workload-class staggered \\
        --num-agents 8 --arrival-rate-per-sec 0.5 \\
        --num-user-prompts 4 --tool-latency-ms 2000 \\
        --concurrency 8

    # Burst:
    PYTHONPATH=. python3 scripts/run_concurrent_eval.py \\
        --config configs/retention_filler.yaml \\
        --output results/burst-$TS/retention/ \\
        --workload-class burst \\
        --num-agents 8 --burst-duration-ms 500 \\
        --num-user-prompts 4 --tool-latency-ms 2000 \\
        --concurrency 8

Output layout (unchanged):
    <output>/<trace_id>.json   per-trace TraceResult dump
    <output>/summary.json      RunSummary + focal/filler split + workload
                                summary (focal-TTFT, p95/p99, Jain, SLO)
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
from evaluation.metrics import RunSummary, TraceResult
from evaluation.run_eval import _configure_retention_events, load_config
from evaluation.trace_loader import (
    fixture_trace,
    filler_focal_workload,
)
from evaluation.workloads import (
    lockstep_workload,
    staggered_workload,
    heterogeneous_workload,
    burst_workload,
)
from evaluation.workload_metrics import WorkloadSummary

WORKLOAD_CHOICES = (
    "legacy",
    "filler_focal",
    "lockstep",
    "staggered",
    "heterogeneous",
    "burst",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a concurrent multi-trace benchmark.")
    p.add_argument(
        "--config", type=Path, required=True,
        help="MoM YAML config (configs/*.yaml).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for per-trace JSON + summary.json.",
    )
    p.add_argument(
        "--concurrency", type=int, default=2,
        help="Max number of in-flight requests at any time.",
    )

    # ---- workload-class selector --------------------------------------------
    p.add_argument(
        "--workload-class", choices=WORKLOAD_CHOICES, default=None,
        help="Workload pattern to build. If omitted, falls back to legacy "
             "(replicate cfg.traces) — UNLESS --num-fillers > 0, in which "
             "case the script auto-selects filler_focal for backward compat.",
    )
    p.add_argument(
        "--slo-threshold-ms", type=float, default=200.0,
        help="TTFT SLO threshold (ms). Used in WorkloadSummary's slo_pass_rate.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for staggered / heterogeneous / burst arrival timing "
             "and tool latency samples. Reproducible across runs.",
    )

    # ---- legacy mode --------------------------------------------------------
    p.add_argument(
        "--num-traces", type=int, default=None,
        help="(legacy) How many replicas of each YAML trace to run. "
             "Defaults to --concurrency.",
    )

    # ---- filler+focal mode --------------------------------------------------
    p.add_argument(
        "--num-fillers", type=int, default=0,
        help="(filler_focal) Number of filler agents. >0 auto-selects "
             "filler_focal even if --workload-class isn't passed.",
    )
    p.add_argument(
        "--focal-num-turns", type=int, default=4,
        help="(filler_focal) Number of user_prompts on the focal trace.",
    )
    p.add_argument(
        "--focal-tool-latency-ms", type=float, default=5000.0,
        help="(filler_focal) Tool gap on focal trace, in ms.",
    )
    p.add_argument(
        "--filler-num-turns", type=int, default=200,
        help="(filler_focal) Per-filler turn count. Should comfortably "
             "exceed what fillers process before focal completes.",
    )
    p.add_argument(
        "--focal-body-tokens", type=int, default=2048,
        help="(filler_focal) Body tokens per focal user_prompt.",
    )
    p.add_argument(
        "--filler-body-tokens", type=int, default=2048,
        help="(filler_focal) Body tokens per filler user_prompt.",
    )

    # ---- general agent flags (shared by lockstep / staggered / etc.) --------
    p.add_argument(
        "--num-agents", type=int, default=8,
        help="(lockstep / staggered / heterogeneous / burst) Number of agents.",
    )
    p.add_argument(
        "--num-user-prompts", type=int, default=4,
        help="(lockstep / staggered / heterogeneous / burst) "
             "User_prompts per agent.",
    )
    p.add_argument(
        "--tool-latency-ms", type=float, default=2000.0,
        help="(lockstep / staggered / burst) Constant tool gap, in ms.",
    )
    p.add_argument(
        "--body-tokens", type=int, default=2048,
        help="(all generic-agent classes) Body tokens per user_prompt.",
    )

    # ---- staggered ---------------------------------------------------------
    p.add_argument(
        "--arrival-rate-per-sec", type=float, default=0.5,
        help="(staggered) Mean Poisson arrival rate. 0.5 = one new agent "
             "every ~2 seconds on average.",
    )

    # ---- heterogeneous -----------------------------------------------------
    p.add_argument(
        "--tool-latency-log-mean-ms", type=float, default=1500.0,
        help="(heterogeneous) Arithmetic mean of the log-normal tool latency "
             "distribution. Internally converted to log-space.",
    )
    p.add_argument(
        "--tool-latency-log-sigma", type=float, default=0.7,
        help="(heterogeneous) Log-space standard deviation. 0.7 ≈ a factor "
             "of 2 spread; higher = more variation.",
    )

    # ---- burst -------------------------------------------------------------
    p.add_argument(
        "--burst-duration-ms", type=float, default=500.0,
        help="(burst) Window during which all agents arrive (uniform).",
    )

    p.add_argument(
        "--mock-engine", action="store_true",
        help="Use evaluation.engine_adapter.MockEngine (no CUDA, smoke testing).",
    )
    return p.parse_args()


def _resolve_workload_class(args: argparse.Namespace) -> str:
    """Pick the workload class from CLI flags, supporting backward-compat
    shortcut (--num-fillers > 0 → filler_focal)."""
    if args.workload_class is not None:
        return args.workload_class
    if args.num_fillers > 0:
        return "filler_focal"
    return "legacy"


def build_specs_legacy(cfg: dict[str, Any], num_traces: int) -> list:
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


def build_specs_filler_focal(cfg: dict[str, Any], args: argparse.Namespace) -> list:
    model = cfg.get("model", {}).get("name", "unknown-model")
    prompt_tokens = (
        cfg.get("model", {}).get("max_model_len")
        or args.focal_body_tokens * args.focal_num_turns
    )
    return filler_focal_workload(
        num_fillers=args.num_fillers,
        focal_num_user_prompts=args.focal_num_turns,
        focal_tool_latency_ms=args.focal_tool_latency_ms,
        filler_num_turns=args.filler_num_turns,
        model=model,
        prompt_tokens=prompt_tokens,
        focal_body_tokens_per_turn=args.focal_body_tokens,
        filler_body_tokens_per_turn=args.filler_body_tokens,
    )


def build_specs_lockstep(cfg: dict[str, Any], args: argparse.Namespace) -> list:
    model = cfg.get("model", {}).get("name", "unknown-model")
    prompt_tokens = cfg.get("model", {}).get("max_model_len", 8192)
    return lockstep_workload(
        num_agents=args.num_agents,
        num_user_prompts=args.num_user_prompts,
        tool_latency_ms=args.tool_latency_ms,
        model=model,
        prompt_tokens=prompt_tokens,
        body_tokens_per_turn=args.body_tokens,
    )


def build_specs_staggered(cfg: dict[str, Any], args: argparse.Namespace) -> list:
    model = cfg.get("model", {}).get("name", "unknown-model")
    prompt_tokens = cfg.get("model", {}).get("max_model_len", 8192)
    return staggered_workload(
        num_agents=args.num_agents,
        arrival_rate_per_sec=args.arrival_rate_per_sec,
        num_user_prompts=args.num_user_prompts,
        tool_latency_ms=args.tool_latency_ms,
        model=model,
        prompt_tokens=prompt_tokens,
        body_tokens_per_turn=args.body_tokens,
        seed=args.seed,
    )


def build_specs_heterogeneous(cfg: dict[str, Any], args: argparse.Namespace) -> list:
    model = cfg.get("model", {}).get("name", "unknown-model")
    prompt_tokens = cfg.get("model", {}).get("max_model_len", 8192)
    return heterogeneous_workload(
        num_agents=args.num_agents,
        num_user_prompts=args.num_user_prompts,
        tool_latency_log_mean_ms=args.tool_latency_log_mean_ms,
        tool_latency_log_sigma=args.tool_latency_log_sigma,
        model=model,
        prompt_tokens=prompt_tokens,
        body_tokens_per_turn=args.body_tokens,
        seed=args.seed,
    )


def build_specs_burst(cfg: dict[str, Any], args: argparse.Namespace) -> list:
    model = cfg.get("model", {}).get("name", "unknown-model")
    prompt_tokens = cfg.get("model", {}).get("max_model_len", 8192)
    return burst_workload(
        num_agents=args.num_agents,
        burst_duration_ms=args.burst_duration_ms,
        num_user_prompts=args.num_user_prompts,
        tool_latency_ms=args.tool_latency_ms,
        model=model,
        prompt_tokens=prompt_tokens,
        body_tokens_per_turn=args.body_tokens,
        seed=args.seed,
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)

    _configure_retention_events(args.output, use_wandb=False)

    workload_class = _resolve_workload_class(args)
    if workload_class == "legacy":
        num_traces = args.num_traces or args.concurrency
        specs = build_specs_legacy(cfg, num_traces=num_traces)
    elif workload_class == "filler_focal":
        if args.num_fillers <= 0:
            raise SystemExit(
                "--workload-class filler_focal requires --num-fillers > 0"
            )
        specs = build_specs_filler_focal(cfg, args)
    elif workload_class == "lockstep":
        specs = build_specs_lockstep(cfg, args)
    elif workload_class == "staggered":
        specs = build_specs_staggered(cfg, args)
    elif workload_class == "heterogeneous":
        specs = build_specs_heterogeneous(cfg, args)
    elif workload_class == "burst":
        specs = build_specs_burst(cfg, args)
    else:
        raise SystemExit(f"unknown workload class: {workload_class}")

    config_name = cfg.get("name", args.config.stem)
    print(f"=== Concurrent run ===")
    print(f"  workload     : {workload_class}")
    print(f"  config       : {config_name}")
    n_focal = sum(1 for s in specs if s.role == "focal")
    n_filler = sum(1 for s in specs if s.role == "filler")
    print(f"  specs        : {len(specs)} (focal={n_focal}, filler={n_filler})")
    print(f"  concurrency  : {args.concurrency}")
    print(f"  output       : {args.output}")
    if any(s.start_offset_ms > 0 for s in specs):
        offsets = sorted(s.start_offset_ms for s in specs)
        print(f"  start_offset : min={offsets[0]:.0f}ms median="
              f"{offsets[len(offsets)//2]:.0f}ms max={offsets[-1]:.0f}ms")
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

    # Legacy RunSummary (kept for back-compat with existing analysis scripts).
    legacy_summary = RunSummary.from_traces(traces)

    # Workload-class-aware summary (the new headline).
    workload_summary = WorkloadSummary.from_traces(
        traces,
        workload_class=workload_class,
        slo_threshold_ms=args.slo_threshold_ms,
    )

    # Combine into one summary.json.
    summary_payload: dict[str, Any] = dict(legacy_summary.__dict__)
    summary_payload["workload"] = workload_summary.to_dict()
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"  wrote {summary_path}")

    print()
    print(workload_summary.format_table())
    print()
    print(f"  one-line: {workload_summary.format_one_line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
