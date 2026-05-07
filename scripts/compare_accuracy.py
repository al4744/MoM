"""Workstream D — Side-by-side accuracy comparison across configs.

Reads ``accuracy.json`` (written by ``run_accuracy_eval.py``) from N config
directories and prints a per-task + headline-mean comparison. Uses the same
``mean_task_accuracy`` extraction as ``run_accuracy_eval.py`` so the numbers
match cell-for-cell.

The point: answer "did our latency optimizations break accuracy?" without
manually opening N JSON files. The headline question is whether
retention_int4 (75% memory savings) holds up against the baseline
accuracy bound.

Usage:
    PYTHONPATH=. python scripts/compare_accuracy.py \\
        results/d-run-20260506-1900/baseline/ \\
        results/d-run-20260506-1900/retention/ \\
        results/d-run-20260506-1900/retention_int8/ \\
        results/d-run-20260506-1900/retention_int4/

Outputs a markdown table to stdout, suitable for paste into the writeup.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "result_dirs", nargs="+", type=Path,
        help="One or more directories containing accuracy.json from "
             "scripts/run_accuracy_eval.py.",
    )
    p.add_argument(
        "--baseline-name", default="baseline",
        help="Config name to use as the diff reference. Other configs' "
             "accuracy is reported as Δ vs this baseline.",
    )
    return p.parse_args()


def _load_accuracy(result_dir: Path) -> dict:
    path = result_dir / "accuracy.json"
    if not path.exists():
        print(f"[!] {path} not found — did run_accuracy_eval.py finish?",
              file=sys.stderr)
        sys.exit(2)
    with open(path) as f:
        return json.load(f)


def _extract_per_task(blob: dict) -> dict[str, float]:
    """Pull (task_name → accuracy) from one accuracy.json blob."""
    out: dict[str, float] = {}
    for task_name, metrics in (blob.get("results") or {}).items():
        # lm-eval uses keys like "acc,none" or "acc_norm,none".
        acc = metrics.get("acc,none")
        if acc is None:
            acc = metrics.get("acc_norm,none")
        if isinstance(acc, (int, float)):
            out[task_name] = float(acc)
    return out


def _mean_acc(per_task: dict[str, float]) -> float:
    if not per_task:
        return 0.0
    return sum(per_task.values()) / len(per_task)


def main() -> int:
    args = parse_args()

    rows: list[tuple[str, dict[str, float], float]] = []
    for d in args.result_dirs:
        blob = _load_accuracy(d)
        config_name = blob.get("config_name") or d.name
        per_task = _extract_per_task(blob)
        rows.append((config_name, per_task, _mean_acc(per_task)))

    # Build the union of tasks across all configs.
    all_tasks = sorted({t for _, per_task, _ in rows for t in per_task})

    # Find baseline row.
    baseline_row = next(
        (r for r in rows if r[0] == args.baseline_name), None
    )
    baseline_acc = baseline_row[2] if baseline_row else None

    # ---- Markdown table -------------------------------------------------
    header = ["config"] + all_tasks + ["mean"]
    if baseline_row is not None:
        header.append("Δ vs baseline")
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for config_name, per_task, mean_acc in rows:
        cells = [config_name]
        for t in all_tasks:
            v = per_task.get(t)
            cells.append(f"{v:.4f}" if v is not None else "—")
        cells.append(f"{mean_acc:.4f}")
        if baseline_row is not None and baseline_acc is not None:
            delta = mean_acc - baseline_acc
            sign = "+" if delta >= 0 else ""
            cells.append(f"{sign}{delta*100:.2f} pp")
        print("| " + " | ".join(cells) + " |")

    # ---- Headline summary ----------------------------------------------
    print()
    print("=== Headline summary ===")
    if baseline_row is None:
        print(f"  no row matched --baseline-name={args.baseline_name!r}; "
              f"showing means only")
        for config_name, _, mean_acc in rows:
            print(f"    {config_name:<25} mean_acc={mean_acc:.4f}")
        return 0

    print(f"  baseline ({args.baseline_name}): mean_acc={baseline_acc:.4f}")
    for config_name, _, mean_acc in rows:
        if config_name == args.baseline_name:
            continue
        delta_pp = (mean_acc - baseline_acc) * 100
        marker = " ⚠" if delta_pp < -2.0 else ""
        print(f"    {config_name:<25} mean_acc={mean_acc:.4f}  "
              f"Δ={delta_pp:+.2f} pp{marker}")
    print()
    print("  ⚠ flag: any config losing >2 pp vs baseline. Memory savings are")
    print("    only worth shipping if the accuracy cost stays under that bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
