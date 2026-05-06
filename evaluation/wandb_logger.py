"""Optional WandB logging for MoM evaluation runs.

This module intentionally has no import-time dependency on wandb. The
evaluation runner imports it only after the normal JSON outputs are written,
and this helper imports wandb lazily so dry-run/mock/offline tests do not need
network access or WANDB_API_KEY.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from evaluation.metrics import RunSummary, TraceResult


SUMMARY_KEYS: tuple[str, ...] = (
    "mean_ttft_ms",
    "mean_prefill_recomp_ms",
    "p99_tbt_ms",
    "mean_peak_vram_mb",
    "total_preemptions",
    "mean_xfer_bandwidth_mb_s",
    "mean_task_accuracy",
)

PER_TURN_COLUMNS: list[str] = [
    "config_name",
    "trace_id",
    "turn_index",
    "tool_name",
    "tool_latency_class",
    "ttft_ms",
    "tbt_ms_mean",
    "tbt_ms_p99",
    "wallclock_ms",
    "prefill_recomp_ms",
    "pin_hits",
    "pin_misses",
    "blocks_reused",
]

OUTPUT_ARTIFACT_SUFFIXES: set[str] = {
    ".json",
    ".md",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".html",
    ".pdf",
}


@dataclass(frozen=True)
class WandbSettings:
    """CLI-derived WandB settings."""

    project: str = "mom-eval"
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: tuple[str, ...] = ()
    mode: str = "online"


def log_evaluation_run(
    *,
    settings: WandbSettings,
    cfg: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    traces: list[TraceResult],
    summary: RunSummary,
    execution_mode: str,
) -> bool:
    """Log an already-completed evaluation run to WandB.

    Returns True if a wandb run was created, False if logging was skipped.
    Missing wandb is a warning, not a failure.
    """
    if settings.mode == "disabled":
        print("[!] wandb mode is disabled; skipping --wandb push", file=sys.stderr)
        return False

    try:
        wandb = importlib.import_module("wandb")
    except ImportError:
        print("[!] wandb not installed; skipping --wandb push", file=sys.stderr)
        return False

    config_name = summary.config_name
    run_name = f"{config_name}-{execution_mode}"
    trace_ids = [trace.trace_id for trace in traces]
    tags = _dedupe_tags((*settings.tags, config_name, execution_mode))
    metadata = _run_metadata(cfg, config_name, trace_ids, execution_mode, traces)

    init_kwargs: dict[str, Any] = {
        "project": settings.project,
        "name": run_name,
        "config": cfg,
        "mode": settings.mode,
        "tags": tags,
        "notes": "MoM inference/system benchmark run; not a training run.",
    }
    if settings.entity:
        init_kwargs["entity"] = settings.entity
    if settings.group:
        init_kwargs["group"] = settings.group

    run = wandb.init(**init_kwargs)
    try:
        wandb.log(_prefixed_summary(summary))
        wandb.log({"run_metadata": metadata})

        per_trace_rows = _per_trace_rows(traces)
        if per_trace_rows:
            _log_table(
                wandb,
                "per_trace_metrics",
                _per_trace_columns(traces),
                per_trace_rows,
            )

        per_turn_rows = _per_turn_rows(traces)
        if per_turn_rows:
            _log_table(wandb, "per_turn_metrics", PER_TURN_COLUMNS, per_turn_rows)

        _log_output_artifact(
            wandb=wandb,
            run=run,
            name=f"{run_name}-outputs",
            output_dir=output_dir,
            config_path=config_path,
        )
    finally:
        wandb.finish()
    return True


def _prefixed_summary(summary: RunSummary) -> dict[str, float | int]:
    blob = asdict(summary)
    out: dict[str, float | int] = {}
    for key in SUMMARY_KEYS:
        value = blob.get(key)
        if value is not None:
            out[f"summary/{key}"] = value

    for key, value in blob.items():
        if value is None or key in SUMMARY_KEYS:
            continue
        if isinstance(value, (int, float)) and (
            "throughput" in key
            or "gpu_util" in key
            or "gpu_utilization" in key
            or key.startswith("mean_quantized_")
            or key == "mean_kv_compression_ratio"
        ):
            out[f"summary/{key}"] = value
    return out


def _run_metadata(
    cfg: dict[str, Any],
    config_name: str,
    trace_ids: list[str],
    execution_mode: str,
    traces: list[TraceResult],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "config_name": config_name,
        "model_name": cfg.get("model", {}).get("name"),
        "trace_ids": trace_ids,
        "execution_mode": execution_mode,
        "dry_run": execution_mode == "dry-run",
        "mock_engine": execution_mode == "mock",
        "real_engine": execution_mode == "real",
    }

    for trace in traces:
        for key in ("timestamp", "timestamp_s", "timestamp_unix_s", "wallclock_ts"):
            if key in trace.metadata:
                metadata[key] = trace.metadata[key]
                return metadata
    return metadata


def _per_trace_columns(traces: list[TraceResult]) -> list[str]:
    columns = [
        "config_name",
        "trace_id",
        "mean_ttft_ms",
        "mean_prefill_recomp_ms",
        "p99_tbt_ms",
        "peak_vram_mb",
        "preemption_count",
        "cpu_gpu_xfer_bytes",
        "cpu_gpu_xfer_ms",
        "cpu_gpu_xfer_bandwidth_mb_s",
        "task_accuracy",
        "quantized_kv_mb",
        "kv_compression_ratio",
    ]
    optional = sorted(
        key
        for trace in traces
        for key in trace.metadata
        if (
            "throughput" in key
            or "gpu_util" in key
            or "gpu_utilization" in key
            or key.startswith("compile_")
            or key.startswith("profile_")
        )
    )
    return columns + [key for key in optional if key not in columns]


def _per_trace_rows(traces: list[TraceResult]) -> list[list[Any]]:
    columns = _per_trace_columns(traces)
    rows: list[list[Any]] = []
    for trace in traces:
        row_values: dict[str, Any] = {
            "config_name": trace.config_name,
            "trace_id": trace.trace_id,
            "mean_ttft_ms": trace.mean_ttft_ms,
            "mean_prefill_recomp_ms": trace.mean_prefill_recomp_ms,
            "p99_tbt_ms": trace.p99_tbt_ms,
            "peak_vram_mb": trace.peak_vram_mb,
            "preemption_count": trace.preemption_count,
            "cpu_gpu_xfer_bytes": trace.cpu_gpu_xfer_bytes,
            "cpu_gpu_xfer_ms": trace.cpu_gpu_xfer_ms,
            "cpu_gpu_xfer_bandwidth_mb_s": trace.cpu_gpu_xfer_bandwidth_mb_s,
            "task_accuracy": trace.task_accuracy,
            "quantized_kv_mb": trace.quantized_kv_mb,
            "kv_compression_ratio": trace.kv_compression_ratio,
        }
        row_values.update(
            {
                key: value
                for key, value in trace.metadata.items()
                if (
                    "throughput" in key
                    or "gpu_util" in key
                    or "gpu_utilization" in key
                    or key.startswith("compile_")
                    or key.startswith("profile_")
                )
            }
        )
        rows.append([row_values.get(column) for column in columns])
    return rows


def _per_turn_rows(traces: list[TraceResult]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for trace in traces:
        for turn in trace.turns:
            rows.append(
                [
                    trace.config_name,
                    trace.trace_id,
                    turn.turn_index,
                    turn.tool_name,
                    turn.tool_latency_class,
                    turn.ttft_ms,
                    turn.tbt_ms_mean,
                    turn.tbt_ms_p99,
                    turn.wallclock_ms,
                    turn.prefill_recomp_ms,
                    turn.pin_hits,
                    turn.pin_misses,
                    turn.blocks_reused,
                ]
            )
    return rows


def _log_table(wandb: Any, key: str, columns: list[str], rows: list[list[Any]]) -> None:
    table = wandb.Table(columns=columns, data=rows)
    wandb.log({key: table})


def _log_output_artifact(
    *,
    wandb: Any,
    run: Any,
    name: str,
    output_dir: Path,
    config_path: Path,
) -> None:
    artifact = wandb.Artifact(name=name, type="evaluation-output")

    if config_path.exists():
        artifact.add_file(str(config_path), name=f"config/{config_path.name}")

    for path in _iter_output_files(output_dir):
        try:
            rel_name = str(path.relative_to(output_dir))
        except ValueError:
            rel_name = f"comparison/{path.name}"
        artifact.add_file(str(path), name=rel_name)

    if hasattr(run, "log_artifact"):
        run.log_artifact(artifact)
    elif hasattr(wandb, "log_artifact"):
        wandb.log_artifact(artifact)


def _iter_output_files(output_dir: Path) -> Iterable[Path]:
    if not output_dir.exists():
        return []

    paths = {
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in OUTPUT_ARTIFACT_SUFFIXES
    }

    for pattern in ("comparison*.md", "ablate*.md"):
        paths.update(path for path in output_dir.parent.glob(pattern) if path.is_file())

    return sorted(paths)


def _dedupe_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for raw_tag in tags:
        tag = raw_tag.strip()
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return tuple(out)
