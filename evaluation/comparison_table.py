"""Workstream D — Markdown comparison table generator.

Two modes:

  1. **Cross-config matrix** (--results-root): one row per config, sorted by
     mean_ttft_ms ascending. Used for the paper's main results table.

  2. **Pairwise ablation** (--ablate): baseline vs candidate, with per-metric
     delta and speedup. Used for "X gives Yx faster TTFT" claims.

Usage:
    # Mode 1 — cross-config table
    python evaluation/comparison_table.py \\
        --results-root results/ \\
        --output results/comparison.md

    # Mode 2 — pairwise delta
    python evaluation/comparison_table.py \\
        --ablate results/baseline/summary.json results/retention/summary.json \\
        --output results/ablate-baseline-vs-retention.md

Cross-config output:

    | Config         | TTFT (ms) | Prefill recomp (ms) | TBT p99 (ms) | ...
    |----------------|-----------|---------------------|--------------|----
    | full-stack     | 121.4     | 11.2                | 12.8         | ...
    | retention-int8 | 162.1     | 12.6                | 14.0         | ...
    | retention      | 165.7     | 12.8                | 13.9         | ...
    | baseline       | 480.2     | 320.5               | 14.1         | ...

Ablation output:

    | Metric              | baseline | retention | Δ      | speedup |
    |---------------------|----------|-----------|--------|---------|
    | TTFT (ms)           | 480.2    | 165.7     | -314.5 | 2.90x   |
    | Prefill recomp (ms) | 320.5    | 12.8      | -307.7 | 25.04x  |
    | TBT p99 (ms)        | 14.1     | 13.9      | -0.2   | 1.01x   |
    | VRAM (MB)           | 38420    | 39100     | +680   | 0.98x   |
    | Preemptions         | 12       | 3         | -9     | 4.00x   |
    | XFER (MB/s)         | 0.0      | 0.0       | 0.0    | —       |
    | Task accuracy       | —        | —         | —      | —       |
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HEADER = (
    "| Config | TTFT (ms) | Prefill recomp (ms) | TBT p99 (ms) "
    "| VRAM (MB) | Preemptions | XFER (MB/s) | Task acc |"
)
DIVIDER = "|" + "|".join("---" for _ in range(8)) + "|"


# Metrics shown in the ablation table, in display order.
# (display_name, summary_key, precision, lower_is_better)
ABLATE_METRICS: tuple[tuple[str, str, int, bool], ...] = (
    ("TTFT (ms)",            "mean_ttft_ms",              1, True),
    ("Prefill recomp (ms)",  "mean_prefill_recomp_ms",    1, True),
    ("TBT p99 (ms)",         "p99_tbt_ms",                1, True),
    ("VRAM (MB)",            "mean_peak_vram_mb",         0, True),
    ("Preemptions",          "total_preemptions",         0, True),
    ("XFER (MB/s)",          "mean_xfer_bandwidth_mb_s",  1, False),
    ("Task accuracy",        "mean_task_accuracy",        3, False),
)


# ---------------------------------------------------------------------------
# Loaders + formatting helpers
# ---------------------------------------------------------------------------

def load_summaries(results_root: Path) -> list[dict[str, Any]]:
    """Walk results_root for `*/summary.json` files and load them."""
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        with open(summary_path) as f:
            summaries.append(json.load(f))
    return summaries


def load_summary(path: Path) -> dict[str, Any]:
    """Load one summary.json file."""
    with open(path) as f:
        return json.load(f)


def fmt(val: float | int | None, prec: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, int):
        return str(val)
    return f"{val:.{prec}f}"


def fmt_delta(baseline: float | int | None, candidate: float | int | None,
              prec: int = 1) -> str:
    """Format the absolute delta with explicit + or - sign."""
    if baseline is None or candidate is None:
        return "—"
    delta = candidate - baseline
    if isinstance(baseline, int) and isinstance(candidate, int):
        return f"{delta:+d}"
    return f"{delta:+.{prec}f}"


def fmt_speedup(baseline: float | int | None, candidate: float | int | None,
                lower_is_better: bool) -> str:
    """Compute speedup ratio.

    - lower_is_better (TTFT, PRT, etc.): speedup = baseline / candidate
    - higher_is_better (bandwidth, accuracy): speedup = candidate / baseline

    Returns em-dash if either value is missing or denominator is zero.
    """
    if baseline is None or candidate is None:
        return "—"
    if lower_is_better:
        if candidate == 0:
            return "—"
        ratio = baseline / candidate
    else:
        if baseline == 0:
            return "—"
        ratio = candidate / baseline
    return f"{ratio:.2f}x"


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------

def emit_table(summaries: list[dict[str, Any]]) -> str:
    """Cross-config comparison table, one row per config."""
    summaries = sorted(summaries, key=lambda s: s.get("mean_ttft_ms", 0.0))

    rows = [HEADER, DIVIDER]
    for s in summaries:
        rows.append(
            "| {name} | {ttft} | {prt} | {tbt} | {vram} | {prmp} | {xfer} | {acc} |".format(
                name=s.get("config_name", "?"),
                ttft=fmt(s.get("mean_ttft_ms")),
                prt=fmt(s.get("mean_prefill_recomp_ms")),
                tbt=fmt(s.get("p99_tbt_ms")),
                vram=fmt(s.get("mean_peak_vram_mb"), 0),
                prmp=fmt(s.get("total_preemptions")),
                xfer=fmt(s.get("mean_xfer_bandwidth_mb_s")),
                acc=fmt(s.get("mean_task_accuracy"), 3),
            )
        )
    return "\n".join(rows) + "\n"


def emit_ablate_table(baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    """Pairwise delta table: baseline vs candidate, with Δ and speedup columns."""
    base_name = baseline.get("config_name", "baseline")
    cand_name = candidate.get("config_name", "candidate")

    header = f"| Metric | {base_name} | {cand_name} | Δ | speedup |"
    divider = "|---|---|---|---|---|"
    rows = [header, divider]

    for display, key, prec, lower_better in ABLATE_METRICS:
        b = baseline.get(key)
        c = candidate.get(key)
        rows.append(
            f"| {display} | {fmt(b, prec)} | {fmt(c, prec)} "
            f"| {fmt_delta(b, c, prec)} | {fmt_speedup(b, c, lower_better)} |"
        )
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit a markdown comparison table.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--results-root",
        type=Path,
        help="Cross-config mode: walk this directory for */summary.json.",
    )
    mode.add_argument(
        "--ablate",
        nargs=2,
        metavar=("BASELINE_SUMMARY", "CANDIDATE_SUMMARY"),
        type=Path,
        help="Pairwise mode: baseline summary.json and candidate summary.json.",
    )
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.results_root is not None:
        summaries = load_summaries(args.results_root)
        if not summaries:
            print(f"[!] No summary.json files under {args.results_root}")
            return 1
        table = emit_table(summaries)
    else:
        baseline = load_summary(args.ablate[0])
        candidate = load_summary(args.ablate[1])
        table = emit_ablate_table(baseline, candidate)

    args.output.write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
