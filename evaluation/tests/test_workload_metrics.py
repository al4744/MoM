"""Tests for evaluation/workload_metrics.py."""
from __future__ import annotations

import json

import pytest

from evaluation.metrics import TraceResult, TurnMetrics
from evaluation.workload_metrics import (
    WorkloadSummary,
    _LatencyDistribution,
    jain_fairness_index,
    quantile,
)


# ---------------------------------------------------------------------------
# Quantile
# ---------------------------------------------------------------------------

class TestQuantile:
    def test_empty_returns_zero(self) -> None:
        assert quantile([], 0.5) == 0.0

    def test_single_element(self) -> None:
        assert quantile([42.0], 0.95) == 42.0

    def test_median_of_odd(self) -> None:
        assert quantile([1, 2, 3, 4, 5], 0.5) == 3.0

    def test_p95_of_uniform(self) -> None:
        # 100 evenly spaced; p95 ≈ index 94 → 95.0 with linear interpolation
        # against rank 0..99 mapping to 1..100.
        xs = list(range(1, 101))
        # k = 99 * 0.95 = 94.05; lo=94, hi=95; lerp 94.05 between xs[94]=95
        # and xs[95]=96 → 95.05.
        assert abs(quantile(xs, 0.95) - 95.05) < 1e-9


# ---------------------------------------------------------------------------
# Jain fairness
# ---------------------------------------------------------------------------

class TestJainFairness:
    def test_perfectly_fair_gives_one(self) -> None:
        assert jain_fairness_index([5.0, 5.0, 5.0, 5.0]) == pytest.approx(1.0)

    def test_one_takes_all_gives_one_over_n(self) -> None:
        # n=4, one trace=4.0, others=0 → (4)^2 / (4 * 16) = 16/64 = 0.25
        assert jain_fairness_index([4.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)

    def test_empty_returns_one(self) -> None:
        assert jain_fairness_index([]) == 1.0

    def test_all_zero_returns_one(self) -> None:
        # all-zero is degenerate; we treat as fair.
        assert jain_fairness_index([0.0, 0.0]) == 1.0


# ---------------------------------------------------------------------------
# _LatencyDistribution
# ---------------------------------------------------------------------------

class TestLatencyDistribution:
    def test_empty_distribution(self) -> None:
        d = _LatencyDistribution.from_values([])
        assert d.n == 0
        assert d.mean_ms == 0.0
        assert d.p95_ms == 0.0

    def test_basic_distribution(self) -> None:
        d = _LatencyDistribution.from_values([10.0, 20.0, 30.0, 40.0, 50.0])
        assert d.n == 5
        assert d.mean_ms == 30.0
        assert d.p50_ms == 30.0
        assert d.max_ms == 50.0


# ---------------------------------------------------------------------------
# WorkloadSummary
# ---------------------------------------------------------------------------

def _make_trace(
    *,
    config_name: str = "test",
    trace_id: str = "t",
    role: str = "focal",
    ttfts: list[float] | None = None,
    is_post_tool: list[bool] | None = None,
    preemptions: int = 0,
) -> TraceResult:
    """Build a TraceResult with N TurnMetrics for unit-testing aggregation."""
    if ttfts is None:
        ttfts = [100.0, 200.0]
    if is_post_tool is None:
        is_post_tool = [False] * len(ttfts)
    assert len(ttfts) == len(is_post_tool)
    tr = TraceResult(config_name=config_name, trace_id=trace_id)
    tr.metadata["role"] = role
    for i, (ttft, post) in enumerate(zip(ttfts, is_post_tool)):
        tr.turns.append(
            TurnMetrics(
                turn_index=i,
                ttft_ms=ttft,
                tbt_ms_mean=5.0,
                tbt_ms_p99=10.0,
                wallclock_ms=ttft + 50.0,
                prefill_recomp_ms=(ttft if post else None),
            )
        )
    tr.preemption_count = preemptions
    return tr


class TestWorkloadSummary:
    def test_requires_at_least_one_trace(self) -> None:
        with pytest.raises(ValueError, match="at least one trace"):
            WorkloadSummary.from_traces([], workload_class="lockstep")

    def test_uniform_config_required(self) -> None:
        traces = [
            _make_trace(config_name="A"),
            _make_trace(config_name="B"),
        ]
        with pytest.raises(ValueError, match="uniform config_name"):
            WorkloadSummary.from_traces(traces, workload_class="lockstep")

    def test_focal_filler_split(self) -> None:
        traces = [
            _make_trace(trace_id="focal-0", role="focal",
                        ttfts=[100.0, 200.0]),
            _make_trace(trace_id="filler-0", role="filler",
                        ttfts=[50.0, 60.0]),
            _make_trace(trace_id="filler-1", role="filler",
                        ttfts=[70.0, 80.0]),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="filler_focal")
        assert s.num_traces == 3
        assert s.num_focal == 1
        assert s.num_filler == 2
        # focal-only TTFTs: [100, 200], mean=150
        assert s.focal_ttft.mean_ms == 150.0
        # filler-only TTFTs: [50, 60, 70, 80], mean=65
        assert s.filler_ttft.mean_ms == 65.0
        # all-trace TTFTs: 6 values, mean=(100+200+50+60+70+80)/6 = 93.33
        assert abs(s.all_ttft.mean_ms - 560.0/6) < 1e-9

    def test_post_tool_only_counts_post_tool_turns(self) -> None:
        # 4 turns, only 2nd & 4th are post-tool.
        traces = [
            _make_trace(role="focal",
                        ttfts=[100.0, 50.0, 200.0, 25.0],
                        is_post_tool=[False, True, False, True]),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        assert s.post_tool_ttft.n == 2
        assert s.post_tool_ttft.mean_ms == (50.0 + 25.0) / 2

    def test_focal_post_tool_excludes_filler(self) -> None:
        traces = [
            _make_trace(trace_id="focal", role="focal",
                        ttfts=[100.0, 25.0],
                        is_post_tool=[False, True]),
            _make_trace(trace_id="filler", role="filler",
                        ttfts=[200.0, 80.0],
                        is_post_tool=[False, True]),  # filler has post_tool too
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="filler_focal")
        # Both contribute to all-post-tool.
        assert s.post_tool_ttft.n == 2
        # Focal-only-post-tool.
        assert s.focal_post_tool_ttft.n == 1
        assert s.focal_post_tool_ttft.mean_ms == 25.0

    def test_jain_fairness_perfect_when_uniform(self) -> None:
        traces = [
            _make_trace(trace_id=f"t{i}", role="focal", ttfts=[100.0, 100.0])
            for i in range(4)
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        assert s.jain_fairness_all == pytest.approx(1.0)
        assert s.jain_fairness_focal == pytest.approx(1.0)

    def test_jain_fairness_drops_under_imbalance(self) -> None:
        # One trace gets 1000ms, three get 100ms each.
        traces = [
            _make_trace(trace_id="slow", role="focal", ttfts=[1000.0]),
            _make_trace(trace_id="t1", role="focal", ttfts=[100.0]),
            _make_trace(trace_id="t2", role="focal", ttfts=[100.0]),
            _make_trace(trace_id="t3", role="focal", ttfts=[100.0]),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        # (1000+300)^2 / (4 * (1e6 + 3*1e4)) = 1.69e6 / 4.12e6 ≈ 0.41
        assert s.jain_fairness_all < 0.5

    def test_slo_pass_rate(self) -> None:
        # SLO=200ms; 3/5 turns under 200.
        traces = [
            _make_trace(role="focal",
                        ttfts=[100.0, 150.0, 199.0, 250.0, 500.0]),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep",
                                        slo_threshold_ms=200.0)
        assert s.slo_pass_rate_all == pytest.approx(3.0 / 5.0)

    def test_total_preemptions_summed(self) -> None:
        traces = [
            _make_trace(trace_id="t0", preemptions=3),
            _make_trace(trace_id="t1", preemptions=7),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        assert s.total_preemptions == 10

    def test_emit_json_round_trip(self, tmp_path) -> None:
        traces = [_make_trace(role="focal", ttfts=[100.0, 200.0])]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        path = tmp_path / "workload.json"
        s.emit_json(path)
        with open(path) as f:
            blob = json.load(f)
        assert blob["workload_class"] == "lockstep"
        assert blob["num_traces"] == 1
        assert "all_ttft" in blob
        assert blob["all_ttft"]["mean_ms"] == 150.0

    def test_format_table_runs(self) -> None:
        # Just check it doesn't throw.
        traces = [
            _make_trace(role="focal", ttfts=[100.0]),
            _make_trace(trace_id="f1", role="filler", ttfts=[200.0]),
        ]
        s = WorkloadSummary.from_traces(traces, workload_class="filler_focal")
        out = s.format_table()
        assert "Workload summary" in out
        assert "post-tool TTFT" in out
        assert "Jain" in out or "jain" in out

    def test_format_one_line(self) -> None:
        traces = [_make_trace(role="focal", ttfts=[100.0])]
        s = WorkloadSummary.from_traces(traces, workload_class="lockstep")
        line = s.format_one_line()
        assert "lockstep" in line
        assert "ttft" in line
