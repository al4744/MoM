"""Automated Workstream C benchmark harness.

This script launches ``evaluation/run_eval.py`` for a context-length sweep with
compile disabled/enabled. It records commands and JSON outputs only; it does
not compute or invent performance conclusions.
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compile benchmark matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--context-lengths", type=int, nargs="+",
                        default=[1024, 4096, 8192, 16384])
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--mock-engine", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="mom-eval")
    parser.add_argument("--wandb-group", default="workstream-c")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required for benchmark harness configs") from exc
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    import yaml  # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(base: dict[str, Any], *, context_length: int,
                 compile_enabled: bool, turns: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    mode = "compile" if compile_enabled else "eager"
    cfg["name"] = f"{base.get('name', 'config')}-{mode}-{context_length}"
    cfg["compile"] = {
        "enabled": compile_enabled,
        "targets": ["prefill", "decode"],
        "backend": "inductor",
        "mode": "default",
        "dynamic": False,
        "fullgraph": False,
        "warmup_iters": 1,
    }
    cfg["profile"] = {
        "enabled": False,
        "pytorch_profiler": False,
        "nsight": False,
        "record_shapes": True,
        "profile_memory": True,
        "with_stack": False,
        "output_dir": str(Path("results/profiles") / cfg["name"]),
    }
    cfg["traces"] = [
        {
            "id": f"{context_length // 1024}k-{turns}turn-mixed",
            "turns": turns,
            "tool_latency_dist": "mixed",
            "prompt_tokens": context_length,
        }
    ]
    return cfg


def main() -> int:
    args = parse_args()
    base = load_yaml(args.config)
    commands_log: list[str] = []

    for context_length in args.context_lengths:
        for compile_enabled in (False, True):
            cfg = build_config(
                base,
                context_length=context_length,
                compile_enabled=compile_enabled,
                turns=args.turns,
            )
            config_path = args.output_root / "configs" / f"{cfg['name']}.yaml"
            output_dir = args.output_root / cfg["name"]
            write_yaml(config_path, cfg)

            command = [
                sys.executable,
                "evaluation/run_eval.py",
                "--config",
                str(config_path),
                "--output",
                str(output_dir),
            ]
            if args.mock_engine:
                command.append("--mock-engine")
            if args.dry_run:
                command.append("--dry-run")
            if args.wandb:
                command.extend([
                    "--wandb",
                    "--wandb-project",
                    args.wandb_project,
                    "--wandb-group",
                    args.wandb_group,
                    "--wandb-tags",
                    "workstream-c,compile" if compile_enabled else "workstream-c,eager",
                ])
            commands_log.append(" ".join(command))
            subprocess.run(command, check=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "commands.txt").write_text("\n".join(commands_log) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
