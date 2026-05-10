"""Workstream D — Open-loop throughput saturation sweep (Path B).

Measures how each engine config (baseline / PC-only / retention) behaves as
arrival rate scales. The headline metric is the *saturation rate at SLO*:
the largest λ (arrival rate, requests/sec) at which the workload's p95
post-tool TTFT stays under a target SLO threshold.

Why this matters: at H100 60-agent batched scale, per-request TTFT is
roughly tied between PC and retention because step time is dominated by
60 concurrent decode operations — saving one focal's prefill is invisible
in the per-request latency framing. But in *open-loop* serving where new
requests keep arriving, the freed compute absorbs the new demand. The
config that frees more compute (retention's pin protecting against
re-prefill on tool returns) sustains a higher arrival rate before its
p95 tail breaks the SLO.

Mechanism: we reuse the existing ``staggered`` workload class — Poisson
arrivals over a fixed agent budget — and sweep ``--arrival-rate-per-sec``
across a configurable grid. For each (config, λ) pair, we run the
workload and record the post-tool TTFT distribution. The saturation
rate is the largest λ where p95 < SLO.

Usage:
    PYTHONPATH=. python scripts/run_throughput_sweep.py \\
        --output results/throughput-sweep-$TS \\
        --slo-ms 500 \\
        --arrival-rates 1,2,4,8,16 \\
        --num-agents 200 \\
        --num-user-prompts 4 \\
        --tool-latency-ms 2000 \\
        --body-tokens 2048

Output:
    <output>/<config>/<rate>/summary.json    per (config, rate) WorkloadSummary
    <output>/saturation_curve.json           the headline analysis: per-config
                                             (rate, p95_ms) tuples + saturation_rate_at_slo
    <output>/saturation_curve.md             markdown table for the writeup
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


CONFIGS = ("baseline_filler", "prefix_cache_only_filler", "retention_filler")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory.")
    p.add_argument("--arrival-rates", default="1,2,4,8,16",
                   help="Comma-separated λ values to sweep (req/sec). "
                        "Default 1,2,4,8,16 covers ~1.5 orders of magnitude.")
    p.add_argument("--slo-ms", type=float, default=500.0,
                   help="p95 post-tool TTFT SLO threshold in ms. Default 500ms.")
    p.add_argument("--num-agents", type=int, default=200,
                   help="Per-rate agent budget. Should comfortably exceed "
                        "λ × test_duration so that the system reaches steady "
                        "state. Default 200.")
    p.add_argument("--num-user-prompts", type=int, default=4,
                   help="Per-agent user_prompts (each followed by a tool gap "
                        "except the last). Default 4 → 3 post-tool turns/agent.")
    p.add_argument("--tool-latency-ms", type=float, default=2000.0,
                   help="Tool gap (ms). Default 2000ms — long enough that "
                        "the focal is paused for measurable cache pressure.")
    p.add_argument("--body-tokens", type=int, default=2048,
                   help="Body tokens per user_prompt. Default 2048.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for Poisson arrival times. Constant across "
                        "configs at a given λ for fair comparison.")
    p.add_argument("--configs", default=",".join(CONFIGS),
                   help="Comma-separated configs to sweep. Default: all three.")
    p.add_argument("--mock-engine", action="store_true",
                   help="Use MockEngine for smoke testing (no GPU).")
    p.add_argument("--python", default=sys.executable,
                   help="Python executable used to spawn the per-rate runs.")
    return p.parse_args()


def run_one(
    *,
    config: str,
    rate: float,
    args: argparse.Namespace,
    output_root: Path,
) -> Path:
    """Run a single (config, λ) point. Returns the summary.json path."""
    out_dir = output_root / config / f"rate-{rate:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python, "scripts/run_concurrent_eval.py",
        "--config", f"configs/{config}.yaml",
        "--output", str(out_dir),
        "--workload-class", "staggered",
        "--num-agents", str(args.num_agents),
        "--num-user-prompts", str(args.num_user_prompts),
        "--tool-latency-ms", str(args.tool_latency_ms),
        "--body-tokens", str(args.body_tokens),
        "--arrival-rate-per-sec", str(rate),
        "--seed", str(args.seed),
        "--concurrency", str(args.num_agents),
        "--slo-threshold-ms", str(args.slo_ms),
    ]
    if args.mock_engine:
        cmd.append("--mock-engine")

    print(f"  → {config} @ {rate:g} RPS", flush=True)
    rc = subprocess.run(cmd, env={"PYTHONPATH": "."} | dict(__import__("os").environ),
                        check=False).returncode
    if rc != 0:
        print(f"    [warn] returncode={rc} — continuing")
    return out_dir / "summary.json"


def collect_curve(output_root: Path, configs: list[str], rates: list[float]) -> dict:
    """Walk the output dirs and pull each (config, rate) → p95 post-tool TTFT."""
    curve: dict[str, list[dict]] = {c: [] for c in configs}
    for cfg in configs:
        for rate in rates:
            summary_path = output_root / cfg / f"rate-{rate:g}" / "summary.json"
            if not summary_path.exists():
                curve[cfg].append({"rate": rate, "p95_ms": None,
                                   "missing": True})
                continue
            with open(summary_path) as f:
                blob = json.load(f)
            workload = blob.get("workload", {})
            post_tool = workload.get("post_tool_ttft", {})
            curve[cfg].append({
                "rate": rate,
                "p95_ms": post_tool.get("p95_ms"),
                "p99_ms": post_tool.get("p99_ms"),
                "mean_ms": post_tool.get("mean_ms"),
                "n": post_tool.get("n"),
                "all_ttft_p95": workload.get("all_ttft", {}).get("p95_ms"),
                "all_ttft_mean": workload.get("all_ttft", {}).get("mean_ms"),
            })
    return curve


def find_saturation(
    points: list[dict], slo_ms: float, metric: str = "p95_ms"
) -> float | None:
    """Largest rate where ``metric`` < ``slo_ms``. Linear search; assumes
    points are sorted by rate ascending and the curve is monotonic."""
    sat = None
    for pt in points:
        v = pt.get(metric)
        if v is None:
            continue
        if v <= slo_ms:
            sat = pt["rate"]
        else:
            break  # monotonic — once we exceed SLO, we stay exceeded
    return sat


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rates = [float(x) for x in args.arrival_rates.split(",")]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]

    print(f"=== Throughput sweep ===")
    print(f"  output       : {args.output}")
    print(f"  configs      : {configs}")
    print(f"  rates (λ)    : {rates}")
    print(f"  SLO          : p95 post-tool TTFT < {args.slo_ms} ms")
    print(f"  agents/run   : {args.num_agents}")
    print(f"  prompts/agent: {args.num_user_prompts}")
    print(f"  tool gap     : {args.tool_latency_ms} ms")
    print(f"  body tokens  : {args.body_tokens}")
    print()

    # Run the full grid
    for cfg in configs:
        print(f"-- {cfg} --")
        for rate in rates:
            run_one(config=cfg, rate=rate, args=args, output_root=args.output)

    # Aggregate
    curve = collect_curve(args.output, configs, rates)
    saturation = {
        cfg: find_saturation(points, args.slo_ms) for cfg, points in curve.items()
    }

    payload = {
        "slo_ms": args.slo_ms,
        "rates_swept": rates,
        "curve": curve,
        "saturation_rate_at_slo": saturation,
        "args": {
            "num_agents": args.num_agents,
            "num_user_prompts": args.num_user_prompts,
            "tool_latency_ms": args.tool_latency_ms,
            "body_tokens": args.body_tokens,
            "seed": args.seed,
        },
    }
    json_path = args.output / "saturation_curve.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    # Markdown table for the report
    md_lines = [
        f"# Throughput sweep — saturation at p95 post-tool TTFT < {args.slo_ms:.0f} ms",
        "",
        "## Saturation rates",
        "",
        "| config | sustained λ at SLO (RPS) |",
        "|---|---|",
    ]
    for cfg, sat in saturation.items():
        sat_str = f"**{sat:g}**" if sat is not None else "—"
        md_lines.append(f"| {cfg} | {sat_str} |")

    md_lines += ["", "## Raw curve (post-tool TTFT)", ""]
    md_lines.append("| config | rate (RPS) | p95 (ms) | p99 (ms) | mean (ms) | n |")
    md_lines.append("|---|---|---|---|---|---|")
    for cfg, points in curve.items():
        for pt in points:
            v95 = pt.get("p95_ms")
            v99 = pt.get("p99_ms")
            vmean = pt.get("mean_ms")
            n = pt.get("n")
            md_lines.append(
                f"| {cfg} | {pt['rate']:g} "
                f"| {v95:.1f} ms" if isinstance(v95, (int, float)) else "| —"
            )
            # Build properly:
            md_lines[-1] = (
                f"| {cfg} | {pt['rate']:g} | "
                + (f"{v95:.1f}" if isinstance(v95, (int, float)) else "—") + " | "
                + (f"{v99:.1f}" if isinstance(v99, (int, float)) else "—") + " | "
                + (f"{vmean:.1f}" if isinstance(vmean, (int, float)) else "—") + " | "
                + (str(n) if n is not None else "—") + " |"
            )
    md_path = args.output / "saturation_curve.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Wrote {md_path}")

    print()
    print("=== Saturation rates at SLO ===")
    for cfg, sat in saturation.items():
        print(f"  {cfg:<28} {sat if sat is not None else '—'} RPS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
