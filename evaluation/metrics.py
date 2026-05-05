"""Workstream D — Metrics suite for tool-aware KV serving.

Defines the six primary metrics from the action plan plus secondary task-accuracy
hooks. Designed to be importable without vLLM as a dependency so that A/B/C can
populate these dataclasses from inside the engine without circular imports.

Primary metrics (all derived from instrumentation in src/retention/events.py
and from PyTorch Profiler / Nsight traces):

  1. TTFT  — time to first token, per turn
  2. TBT   — time between tokens, per turn (mean + p50/p99)
  3. PRT   — prefill recomputation time after tool return, per post-tool turn
  4. VRAM  — peak GPU memory utilization, per trace
  5. PRMP  — vLLM preemption count, per trace
  6. XFER  — CPU↔GPU transfer time + bandwidth, per trace

Secondary metrics (realism layers):
  - AgentBench task accuracy
  - ToolBench task accuracy

Usage from inside an engine run loop:

    from evaluation.metrics import TimingContext, TurnMetrics, TraceResult

    trace = TraceResult(config_name="retention", trace_id="50turn-mixed")
    for turn in trace_iterator:
        with TimingContext() as t:
            ttft = run_prefill(turn)
            tbts = run_decode(turn)
        trace.turns.append(TurnMetrics(
            turn_index=turn.idx,
            ttft_ms=ttft,
            tbt_ms_mean=mean(tbts),
            tbt_ms_p99=p99(tbts),
            prefill_recomp_ms=t.elapsed_ms if turn.post_tool else None,
            wallclock_ms=t.elapsed_ms,
        ))
    trace.peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    trace.preemption_count = engine.scheduler.metrics.num_preemptions
    trace.emit_json("results/retention-50turn.json")
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Timing primitive
# ---------------------------------------------------------------------------

class TimingContext:
    """Context manager for measuring wall-clock latency in milliseconds.

    Uses time.monotonic_ns to align with src/retention/events.py timestamps.
    Nestable; each instance carries its own start/elapsed.
    """

    __slots__ = ("_start_ns", "elapsed_ms")

    def __init__(self) -> None:
        self._start_ns: int = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start_ns = time.monotonic_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ms = (time.monotonic_ns() - self._start_ns) / 1_000_000


# ---------------------------------------------------------------------------
# Per-turn metrics
# ---------------------------------------------------------------------------

@dataclass
class TurnMetrics:
    """Metrics captured for a single conversation turn.

    A turn is one (user-prompt → tool-pause → resume → tokens) cycle. Post-tool
    turns are the ones where prefill recomputation overhead actually matters.
    """

    turn_index: int
    ttft_ms: float
    tbt_ms_mean: float
    tbt_ms_p99: float
    wallclock_ms: float

    # Only populated for turns immediately following a tool return.
    prefill_recomp_ms: Optional[float] = None

    # Tool-call metadata, useful for stratified analysis.
    tool_name: Optional[str] = None
    tool_latency_class: Optional[str] = None  # "fast" | "medium" | "long"

    # Per-turn pin/cache events from src/retention/events.py.
    pin_hits: int = 0
    pin_misses: int = 0
    blocks_reused: int = 0


# ---------------------------------------------------------------------------
# Per-trace results
# ---------------------------------------------------------------------------

@dataclass
class TraceResult:
    """All metrics from one (config, trace) pair.

    `config_name` matches a key in configs/ (e.g. "baseline", "retention",
    "retention-int8", "full-stack"). `trace_id` matches a synthetic trace
    descriptor (e.g. "50turn-mixed-latency").
    """

    config_name: str
    trace_id: str

    # Per-turn breakdown.
    turns: list[TurnMetrics] = field(default_factory=list)

    # Per-trace aggregates (populated post-run).
    peak_vram_mb: Optional[float] = None
    preemption_count: int = 0
    cpu_gpu_xfer_bytes: int = 0
    cpu_gpu_xfer_ms: float = 0.0

    # Optional accuracy metric for AgentBench/ToolBench traces.
    task_accuracy: Optional[float] = None

    # Free-form notes from the runner (model name, GPU type, git SHA, etc.).
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties — lazy, consumed by comparison_table.py
    # ------------------------------------------------------------------

    @property
    def cpu_gpu_xfer_bandwidth_mb_s(self) -> float:
        if self.cpu_gpu_xfer_ms == 0:
            return 0.0
        return (self.cpu_gpu_xfer_bytes / 1e6) / (self.cpu_gpu_xfer_ms / 1e3)

    @property
    def post_tool_turns(self) -> list[TurnMetrics]:
        return [t for t in self.turns if t.prefill_recomp_ms is not None]

    @property
    def mean_ttft_ms(self) -> float:
        return statistics.mean(t.ttft_ms for t in self.turns) if self.turns else 0.0

    @property
    def mean_prefill_recomp_ms(self) -> float:
        post = self.post_tool_turns
        return statistics.mean(t.prefill_recomp_ms for t in post) if post else 0.0

    @property
    def p99_tbt_ms(self) -> float:
        if not self.turns:
            return 0.0
        return max(t.tbt_ms_p99 for t in self.turns)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def emit_json(self, path: str | Path) -> None:
        """Write self as a single JSON document at `path`."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "TraceResult":
        with open(path) as f:
            blob = json.load(f)
        turns = [TurnMetrics(**t) for t in blob.pop("turns", [])]
        return cls(turns=turns, **blob)


# ---------------------------------------------------------------------------
# Run-level aggregate (across multiple traces of the same config)
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """Aggregate of N TraceResults sharing the same config_name."""

    config_name: str
    num_traces: int

    mean_ttft_ms: float
    mean_prefill_recomp_ms: float
    p99_tbt_ms: float
    mean_peak_vram_mb: float
    total_preemptions: int
    mean_xfer_bandwidth_mb_s: float

    mean_task_accuracy: Optional[float] = None

    @classmethod
    def from_traces(cls, traces: list[TraceResult]) -> "RunSummary":
        if not traces:
            raise ValueError("RunSummary requires at least one trace")
        config_names = {t.config_name for t in traces}
        if len(config_names) != 1:
            raise ValueError(
                f"RunSummary requires uniform config_name; got {config_names}"
            )

        accs = [t.task_accuracy for t in traces if t.task_accuracy is not None]
        return cls(
            config_name=traces[0].config_name,
            num_traces=len(traces),
            mean_ttft_ms=statistics.mean(t.mean_ttft_ms for t in traces),
            mean_prefill_recomp_ms=statistics.mean(
                t.mean_prefill_recomp_ms for t in traces
            ),
            p99_tbt_ms=max(t.p99_tbt_ms for t in traces),
            mean_peak_vram_mb=statistics.mean(
                t.peak_vram_mb or 0.0 for t in traces
            ),
            total_preemptions=sum(t.preemption_count for t in traces),
            mean_xfer_bandwidth_mb_s=statistics.mean(
                t.cpu_gpu_xfer_bandwidth_mb_s for t in traces
            ),
            mean_task_accuracy=(statistics.mean(accs) if accs else None),
        )
