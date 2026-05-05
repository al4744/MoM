"""Workstream D — Markdown comparison table generator.

Reads a directory of `summary.json` files (one per config) and emits a single
markdown table comparing all primary metrics across configurations.

Usage:
    python evaluation/comparison_table.py \\
        --results-root results/ \\
        --output results/comparison.md

Produces output like:

    | Config             | TTFT (ms) | Prefill recomp (ms) | TBT p99 (ms) | VRAM (MB) | Preemptions | XFER (MB/s) | Acc |
    |--------------------|-----------|---------------------|--------------|-----------|-------------|-------------|-----|
    | baseline           | 480.2     | 320.5               | 14.1         | 38420     | 12          | 0.0         | —   |
    | retention          | 165.7     | 12.8                | 13.9         | 39100     | 3           | 0.0         | —   |
    | retention-int8     | 162.1     | 12.6                | 14.0         | 28350     | 1           | 0.0         | 0.97|
    | full-stack         | 121.4     | 11.2                | 12.8         | 28200     | 1           | 0.0         | 0.96|

Configs are sorted by `mean_ttft_ms` ascending so the best stack rises to the top.
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


def load_summaries(results_root: Path) -> list[dict[str, Any]]:
    """Walk results_root for `*/summary.json` files and load them."""
    summaries: list[dict[str, Any]] = []
    for summary_path in sorted(results_root.glob("*/summary.json")):
        with open(summary_path) as f:
            summaries.append(json.load(f))
    return summaries


def fmt(val: float | int | None, prec: int = 1) -> str:
    if val is None:
        return "—"
    if isinstance(val, int):
        return str(val)
    return f"{val:.{prec}f}"


def emit_table(summaries: list[dict[str, Any]]) -> str:
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emit comparison table across eval runs.")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    summaries = load_summaries(args.results_root)
    if not summaries:
        print(f"[!] No summary.json files under {args.results_root}")
        return 1

    table = emit_table(summaries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
