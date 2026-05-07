"""Workstream D — Accuracy benchmark via lm-evaluation-harness (Option B).

Runs an lm-eval task suite against a config-specified vLLM engine. Reuses the
same retention / quantization / prefix-caching wiring as
``evaluation/run_eval.py`` so the accuracy numbers match the latency
numbers cell-for-cell in the eval matrix.

The point of this script is to answer ONE question: did our optimizations
(retention, INT8 KV, INT4 KV, prefix-caching) break the model's ability to
reason and follow instructions? If retention_int4 hits 90% MMLU vs
baseline's 92%, we've quantified the accuracy cost of memory savings.

Outputs:
  - ``<output>/accuracy.json``     — full lm-eval results dump (per-task)
  - ``<output>/summary.json``      — updated in-place with mean_task_accuracy
                                     (only if the file already exists)

Usage:
    # Single task (existing):
    PYTHONPATH=. python scripts/run_accuracy_eval.py \\
        --config configs/baseline.yaml \\
        --output results/d-run-20260506-1900/baseline/ \\
        --tasks mmlu --limit 50

    # Task suite shortcut:
    PYTHONPATH=. python scripts/run_accuracy_eval.py \\
        --config configs/retention.yaml \\
        --output results/.../retention/ \\
        --task-suite reasoning --limit 100

    # Full agentic suite (proxies for AgentBench / ToolBench reasoning):
    PYTHONPATH=. python scripts/run_accuracy_eval.py \\
        --config configs/retention_int8.yaml \\
        --output results/.../retention_int8/ \\
        --task-suite agentic --limit 50

Task suites (selected via --task-suite, mutually exclusive with --tasks):

    mmlu          mmlu                     — knowledge breadth (the default).
    reasoning     gsm8k,bbh                — math word problems + BIG-Bench
                                             Hard. Multi-step reasoning IS
                                             the substrate of tool-use, so a
                                             retention/quant config that
                                             fails reasoning will fail
                                             agentic tasks too.
    agentic       mmlu,gsm8k,bbh,arc_challenge
                                           — combined reasoning + knowledge,
                                             our practical proxy for full
                                             AgentBench / ToolBench. Heavier
                                             benchmarks like AgentBench require
                                             external tool environments not
                                             available in this script; pass
                                             them via --tasks if you have
                                             lm-eval task plugins for them
                                             registered locally.

A note on AgentBench / ToolBench specifically:
  Both are multi-turn tool-execution benchmarks requiring a running tool
  environment (DB shells, web sandboxes, etc.). lm-eval-harness does not
  ship those harnesses by default — they need separate setup. Once you have
  an lm-eval-compatible task plugin for AgentBench or ToolBench installed,
  pass its task name via ``--tasks`` and this script forwards it through
  unchanged. As a practical proxy without that infrastructure, the
  ``agentic`` task suite covers the underlying reasoning capabilities the
  agentic benchmarks exercise.

Notes on integration with Workstream A's RetentionConfig:
    lm-eval's ``vllm`` backend builds its own ``LLM(...)`` instance and only
    forwards string-parseable kwargs.  Daksh's ``retention_config`` is a
    Python dataclass, not string-passable.  We work around this by patching
    ``VLLM.__init__`` to inject ``retention_config`` before instantiation.
    The patch is scoped to this single CLI invocation; importing this module
    has no global side effects.

Time budget (Llama 3 8B on 1×L4):
    --limit 20  per-subtask  ≈   3 min/config  → 15 min for 5 configs
    --limit 50  per-subtask  ≈   8 min/config  → 40 min for 5 configs
    --limit 200 per-subtask  ≈  30 min/config  →  2.5 h for 5 configs
    no limit    (full MMLU)  ≈  90 min/config  →  7.5 h for 5 configs
    --task-suite reasoning --limit 50  ≈  20 min/config  → 1.5 h for 5 configs
    --task-suite agentic   --limit 50  ≈  35 min/config  → 3 h for 5 configs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _build_vllm_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate a MoM YAML config into the kwargs lm-eval's VLLM accepts."""
    model_cfg = cfg.get("model", {})
    engine_cfg = cfg.get("engine", {})

    kwargs: dict[str, Any] = {
        "pretrained": model_cfg.get("name"),
        "dtype": model_cfg.get("dtype", "auto"),
        "max_model_len": model_cfg.get("max_model_len", 8192),
        "gpu_memory_utilization": model_cfg.get("gpu_memory_utilization", 0.9),
    }

    if engine_cfg.get("prefix_caching", {}).get("enabled"):
        kwargs["enable_prefix_caching"] = True

    quant = engine_cfg.get("quantization", {}).get("kv_cache")
    if quant:
        kwargs["kv_cache_dtype"] = quant

    return kwargs


def _maybe_inject_retention_config(cfg: dict[str, Any]) -> None:
    """Patch lm_eval's VLLM class to forward retention_config kwarg.

    No-op when retention is disabled.  When enabled, builds a RetentionConfig
    via the same helper run_eval.py uses, then monkey-patches VLLM.__init__
    so retention_config flows into LLM(...) at construction time.
    """
    from evaluation.engine_adapter import build_retention_config

    retention_dict = cfg.get("engine", {}).get("retention", {})
    retention_config = build_retention_config(retention_dict)
    if retention_config is None:
        return

    # Lazy import to avoid hitting CUDA at module load.
    from lm_eval.models.vllm_causallms import VLLM as _VLLM

    original_init = _VLLM.__init__

    def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("retention_config", retention_config)
        return original_init(self, *args, **kwargs)

    _VLLM.__init__ = _patched_init  # type: ignore[method-assign]


def _extract_mean_accuracy(results: dict[str, Any]) -> float | None:
    """Return the unweighted mean of all ``acc`` / ``acc,none`` metrics."""
    accs: list[float] = []
    for _task_name, metrics in (results.get("results") or {}).items():
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            # lm_eval keys look like "acc,none" or "acc_norm,none".
            head = metric_name.split(",")[0]
            if head in {"acc", "acc_norm"}:
                accs.append(float(value))
    if not accs:
        return None
    return sum(accs) / len(accs)


def _update_summary_json(summary_path: Path, mean_acc: float) -> None:
    """In-place update of summary.json with mean_task_accuracy. No-op if absent."""
    if not summary_path.exists():
        return
    with open(summary_path) as f:
        summary = json.load(f)
    summary["mean_task_accuracy"] = mean_acc
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


TASK_SUITES: dict[str, str] = {
    "mmlu":       "mmlu",
    "reasoning":  "gsm8k,bbh",
    "agentic":    "mmlu,gsm8k,bbh,arc_challenge",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="MoM config YAML (configs/*.yaml).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dir (typically the same dir as the matching summary.json).",
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated lm-eval tasks. Examples: mmlu, hellaswag, "
             "arc_easy, arc_challenge, gsm8k, bbh. If you have an "
             "AgentBench/ToolBench lm-eval plugin installed, pass its task "
             "name(s) here.",
    )
    grp.add_argument(
        "--task-suite",
        choices=tuple(TASK_SUITES.keys()),
        default=None,
        help="Predefined task collection. mmlu (knowledge), reasoning "
             "(gsm8k+bbh), agentic (combined — proxy for AgentBench/ToolBench "
             "reasoning subset).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per (sub)task. None = full benchmark. "
             "MMLU full ≈ 90 min/config on L4; --limit 50 ≈ 8 min/config.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="lm-eval batch size. Higher = faster, more VRAM.",
    )
    args = p.parse_args()
    # Default to MMLU if neither flag is given (preserves prior behaviour).
    if args.tasks is None and args.task_suite is None:
        args.tasks = TASK_SUITES["mmlu"]
        args._tasks_origin = "default-mmlu"
    elif args.task_suite is not None:
        args.tasks = TASK_SUITES[args.task_suite]
        args._tasks_origin = f"suite:{args.task_suite}"
    else:
        args._tasks_origin = "explicit"
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(args.config)
    vllm_kwargs = _build_vllm_kwargs(cfg)
    _maybe_inject_retention_config(cfg)

    print(f"=== {args.config.stem} ===")
    print(f"  tasks   : {args.tasks}  ({getattr(args, '_tasks_origin', '?')})")
    print(f"  limit   : {args.limit}")
    print(f"  batch   : {args.batch_size}")
    print(f"  vllm    : {vllm_kwargs}")

    # Convert kwargs to the comma-string lm-eval expects.
    model_args_str = ",".join(f"{k}={v}" for k, v in vllm_kwargs.items())

    from lm_eval import evaluator

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    results = evaluator.simple_evaluate(
        model="vllm",
        model_args=model_args_str,
        tasks=task_list,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    # Persist full per-task results.
    accuracy_path = args.output / "accuracy.json"
    payload = {
        "config_name": cfg.get("name", args.config.stem),
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "results": results.get("results", {}),
        "config_path": str(args.config),
    }
    with open(accuracy_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  wrote {accuracy_path}")

    # Compute headline number and update summary.json (if present).
    mean_acc = _extract_mean_accuracy(results)
    if mean_acc is not None:
        _update_summary_json(args.output / "summary.json", mean_acc)
        print(f"  mean_task_accuracy = {mean_acc:.4f}")
    else:
        print("  [warn] no acc/acc_norm metrics found in results")

    # Per-task table to stdout.
    print("\n=== Per-task accuracy ===")
    for task_name, metrics in (results.get("results") or {}).items():
        acc = metrics.get("acc,none") or metrics.get("acc_norm,none")
        if isinstance(acc, (int, float)):
            print(f"  {task_name}: {acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
