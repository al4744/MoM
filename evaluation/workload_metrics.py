"""Workstream D — Per-workload-class metric aggregation.

The default ``RunSummary`` from ``evaluation.metrics`` averages across all
traces in a run, which loses information when the workload mixes roles
(focal vs filler) or when we care about the tail (p95 / p99) rather than
the mean.

``WorkloadSummary`` here computes the right cross-trace aggregations for
the comprehensive battery:

  * mean / p50 / p95 / p99 of all-trace TTFT
  * mean / p50 / p95 / p99 of post-tool TTFT (the headline retention number)
  * focal-only and filler-only splits when role metadata is set
  * Jain's fairness index over per-trace mean TTFT (1.0 = all agents see
    identical service; lower = some agents starve)
  * throughput-at-SLO: fraction of (turn) requests under a TTFT threshold

Design intent: this dataclass is the single source of truth for the
"headline number" the comparison_table.py / paper plots consume. New
metrics get added here, not scattered in the runner.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

from evaluation.metrics import TraceResult, TurnMetrics


# ---------------------------------------------------------------------------
# Quantile / fairness helpers
# ---------------------------------------------------------------------------

def quantile(xs: Iterable[float], q: float) -> float:
    """Linear-interpolation quantile (matches numpy.quantile default).

    For empty input returns 0.0 (rather than raising) — the comparison
    table treats "no data" rows uniformly with this convention.
    """
    s = sorted(xs)
    if not s:
        return 0.0
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def jain_fairness_index(values: list[float]) -> float:
    """Jain's fairness index over a list of per-trace mean TTFTs.

    F = (Σ x)² / (n · Σ x²)  ∈ [1/n, 1]

    1.0 = every trace sees the exact same mean TTFT.
    1/n = one trace gets all the resources, others see 0.

    A retention configuration that pins one focal agent's blocks at the
    expense of fillers' blocks should show LOWER fairness — that's the
    intended trade-off when the focal trace is "important".
    """
    n = len(values)
    if n == 0:
        return 1.0
    s = sum(values)
    sq = sum(v * v for v in values)
    if sq == 0:
        return 1.0  # all zeros — nominally fair
    return (s * s) / (n * sq)


# ---------------------------------------------------------------------------
# WorkloadSummary
# ---------------------------------------------------------------------------

@dataclass
class _LatencyDistribution:
    """One TTFT distribution's summary stats."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @classmethod
    def from_values(cls, values: list[float]) -> "_LatencyDistribution":
        if not values:
            return cls(n=0, mean_ms=0.0, p50_ms=0.0, p95_ms=0.0, p99_ms=0.0, max_ms=0.0)
        return cls(
            n=len(values),
            mean_ms=statistics.mean(values),
            p50_ms=quantile(values, 0.50),
            p95_ms=quantile(values, 0.95),
            p99_ms=quantile(values, 0.99),
            max_ms=max(values),
        )


@dataclass
class WorkloadSummary:
    """All headline numbers for one (config, workload-class) run."""

    workload_class: str
    config_name: str
    num_traces: int
    num_focal: int
    num_filler: int

    # All-trace TTFT distribution (every user_prompt turn from every trace).
    all_ttft: _LatencyDistribution
    # Post-tool TTFT (TURNS following a tool_return — the headline retention
    # number; this is where pin-or-evict pays off).
    post_tool_ttft: _LatencyDistribution

    # Focal-only / filler-only TTFT splits (zero values when num_focal /
    # num_filler is 0).
    focal_ttft: _LatencyDistribution
    focal_post_tool_ttft: _LatencyDistribution
    filler_ttft: _LatencyDistribution

    # Jain's fairness over per-trace mean TTFT.
    jain_fairness_all: float
    jain_fairness_focal: float

    # Throughput at SLO — fraction of turns whose TTFT is under a threshold.
    # The threshold defaults to 200 ms (a typical "interactive" SLO target);
    # set via from_traces(..., slo_threshold_ms=...).
    slo_threshold_ms: float
    slo_pass_rate_all: float
    slo_pass_rate_focal: float
    slo_pass_rate_post_tool: float

    # Total preemptions across all traces (vLLM scheduler counter).
    total_preemptions: int

    @classmethod
    def from_traces(
        cls,
        traces: list[TraceResult],
        *,
        workload_class: str,
        slo_threshold_ms: float = 200.0,
    ) -> "WorkloadSummary":
        """Aggregate per-trace results into one workload-level summary."""
        if not traces:
            raise ValueError("WorkloadSummary requires at least one trace")
        config_names = {t.config_name for t in traces}
        if len(config_names) != 1:
            raise ValueError(
                f"WorkloadSummary requires uniform config_name; got {config_names}"
            )

        focal_traces = [t for t in traces if t.metadata.get("role", "focal") == "focal"]
        filler_traces = [t for t in traces if t.metadata.get("role") == "filler"]

        all_ttft = [tm.ttft_ms for t in traces for tm in t.turns]
        post_tool_ttft = [
            tm.ttft_ms for t in traces for tm in t.turns
            if tm.prefill_recomp_ms is not None
        ]
        focal_ttft = [tm.ttft_ms for t in focal_traces for tm in t.turns]
        focal_post_tool_ttft = [
            tm.ttft_ms for t in focal_traces for tm in t.turns
            if tm.prefill_recomp_ms is not None
        ]
        filler_ttft = [tm.ttft_ms for t in filler_traces for tm in t.turns]

        per_trace_mean = [t.mean_ttft_ms for t in traces if t.turns]
        per_focal_mean = [t.mean_ttft_ms for t in focal_traces if t.turns]

        def _slo_rate(values: list[float]) -> float:
            if not values:
                return 0.0
            return sum(1 for v in values if v <= slo_threshold_ms) / len(values)

        return cls(
            workload_class=workload_class,
            config_name=traces[0].config_name,
            num_traces=len(traces),
            num_focal=len(focal_traces),
            num_filler=len(filler_traces),
            all_ttft=_LatencyDistribution.from_values(all_ttft),
            post_tool_ttft=_LatencyDistribution.from_values(post_tool_ttft),
            focal_ttft=_LatencyDistribution.from_values(focal_ttft),
            focal_post_tool_ttft=_LatencyDistribution.from_values(focal_post_tool_ttft),
            filler_ttft=_LatencyDistribution.from_values(filler_ttft),
            jain_fairness_all=jain_fairness_index(per_trace_mean),
            jain_fairness_focal=jain_fairness_index(per_focal_mean),
            slo_threshold_ms=slo_threshold_ms,
            slo_pass_rate_all=_slo_rate(all_ttft),
            slo_pass_rate_focal=_slo_rate(focal_ttft),
            slo_pass_rate_post_tool=_slo_rate(post_tool_ttft),
            total_preemptions=sum(t.preemption_count for t in traces),
        )

    # ------------------------------------------------------------------
    # I/O & pretty-print
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialisable form suitable for json.dump."""
        return asdict(self)

    def emit_json(self, path: str | Path) -> None:
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def format_one_line(self) -> str:
        """Single-line summary for stdout / logs."""
        return (
            f"workload={self.workload_class} config={self.config_name} "
            f"n={self.num_traces} (focal={self.num_focal}, filler={self.num_filler}) "
            f"ttft.mean={self.all_ttft.mean_ms:.1f}ms "
            f"post_tool.p95={self.post_tool_ttft.p95_ms:.1f}ms "
            f"focal_post_tool.mean={self.focal_post_tool_ttft.mean_ms:.1f}ms "
            f"jain={self.jain_fairness_all:.3f} "
            f"slo@{self.slo_threshold_ms:.0f}ms={self.slo_pass_rate_all*100:.1f}%"
        )

    def format_table(self) -> str:
        """Multi-line block for stdout. Useful for the per-config breakdown."""
        lines = [
            f"=== Workload summary ===",
            f"  workload_class : {self.workload_class}",
            f"  config         : {self.config_name}",
            f"  traces         : {self.num_traces} (focal={self.num_focal}, filler={self.num_filler})",
            f"",
            f"  --- TTFT (all turns, ms) ---",
            f"     n={self.all_ttft.n:>4}  mean={self.all_ttft.mean_ms:>8.2f}  "
            f"p50={self.all_ttft.p50_ms:>8.2f}  p95={self.all_ttft.p95_ms:>8.2f}  "
            f"p99={self.all_ttft.p99_ms:>8.2f}  max={self.all_ttft.max_ms:>8.2f}",
            f"",
            f"  --- post-tool TTFT (HEADLINE for retention vs PC, ms) ---",
            f"     n={self.post_tool_ttft.n:>4}  mean={self.post_tool_ttft.mean_ms:>8.2f}  "
            f"p50={self.post_tool_ttft.p50_ms:>8.2f}  p95={self.post_tool_ttft.p95_ms:>8.2f}  "
            f"p99={self.post_tool_ttft.p99_ms:>8.2f}  max={self.post_tool_ttft.max_ms:>8.2f}",
        ]
        if self.num_focal > 0:
            lines += [
                f"",
                f"  --- focal-only TTFT (ms) ---",
                f"     n={self.focal_ttft.n:>4}  mean={self.focal_ttft.mean_ms:>8.2f}  "
                f"p50={self.focal_ttft.p50_ms:>8.2f}  p95={self.focal_ttft.p95_ms:>8.2f}  "
                f"p99={self.focal_ttft.p99_ms:>8.2f}",
                f"  --- focal post-tool TTFT (ms) ---",
                f"     n={self.focal_post_tool_ttft.n:>4}  "
                f"mean={self.focal_post_tool_ttft.mean_ms:>8.2f}  "
                f"p50={self.focal_post_tool_ttft.p50_ms:>8.2f}  "
                f"p95={self.focal_post_tool_ttft.p95_ms:>8.2f}  "
                f"p99={self.focal_post_tool_ttft.p99_ms:>8.2f}",
            ]
        if self.num_filler > 0:
            lines += [
                f"",
                f"  --- filler-only TTFT (ms) ---",
                f"     n={self.filler_ttft.n:>4}  mean={self.filler_ttft.mean_ms:>8.2f}  "
                f"p50={self.filler_ttft.p50_ms:>8.2f}  p95={self.filler_ttft.p95_ms:>8.2f}  "
                f"p99={self.filler_ttft.p99_ms:>8.2f}",
            ]
        lines += [
            f"",
            f"  --- fairness & SLO ---",
            f"     jain (all)        : {self.jain_fairness_all:.4f}",
            f"     jain (focal)      : {self.jain_fairness_focal:.4f}",
            f"     SLO @ {self.slo_threshold_ms:.0f} ms        : "
            f"{self.slo_pass_rate_all*100:>5.1f}% all  | "
            f"{self.slo_pass_rate_focal*100:>5.1f}% focal | "
            f"{self.slo_pass_rate_post_tool*100:>5.1f}% post-tool",
            f"     total preemptions : {self.total_preemptions}",
        ]
        return "\n".join(lines)
