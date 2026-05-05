"""Tests for evaluation/metrics.py."""
from __future__ import annotations

import json
import time

import pytest

from evaluation.metrics import (
    RunSummary,
    TimingContext,
    TraceResult,
    TurnMetrics,
)


# ---------------------------------------------------------------------------
# TimingContext
# ---------------------------------------------------------------------------

class TestTimingContext:
    def test_measures_elapsed_ms(self) -> None:
        with TimingContext() as t:
            time.sleep(0.005)
        # Sleep was 5ms; allow generous slack on CI / mac.
        assert 4.0 < t.elapsed_ms < 200.0

    def test_zero_before_exit(self) -> None:
        t = TimingContext()
        assert t.elapsed_ms == 0.0

    def test_nestable(self) -> None:
        with TimingContext() as outer:
            with TimingContext() as inner:
                time.sleep(0.001)
        assert inner.elapsed_ms > 0
        assert outer.elapsed_ms >= inner.elapsed_ms


# ---------------------------------------------------------------------------
# TurnMetrics
# ---------------------------------------------------------------------------

class TestTurnMetrics:
    def test_minimal_construction(self) -> None:
        m = TurnMetrics(
            turn_index=0,
            ttft_ms=120.0,
            tbt_ms_mean=12.0,
            tbt_ms_p99=18.0,
            wallclock_ms=300.0,
        )
        assert m.prefill_recomp_ms is None
        assert m.tool_name is None
        assert m.pin_hits == 0

    def test_with_tool_metadata(self) -> None:
        m = TurnMetrics(
            turn_index=3,
            ttft_ms=80.0,
            tbt_ms_mean=10.0,
            tbt_ms_p99=14.0,
            wallclock_ms=200.0,
            prefill_recomp_ms=15.5,
            tool_name="search",
            tool_latency_class="medium",
            pin_hits=2,
            blocks_reused=128,
        )
        assert m.tool_name == "search"
        assert m.pin_hits == 2


# ---------------------------------------------------------------------------
# TraceResult
# ---------------------------------------------------------------------------

def _make_turns(n: int, post_tool_every: int = 3) -> list[TurnMetrics]:
    return [
        TurnMetrics(
            turn_index=i,
            ttft_ms=100.0 + i,
            tbt_ms_mean=10.0,
            tbt_ms_p99=15.0 + (i % 5),
            wallclock_ms=200.0,
            prefill_recomp_ms=(50.0 if i > 0 and i % post_tool_every == 0 else None),
            tool_name=("search" if i % post_tool_every == 0 else None),
        )
        for i in range(n)
    ]


class TestTraceResult:
    def test_post_tool_turns_filter(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=_make_turns(10, post_tool_every=3),
        )
        # Indices 3, 6, 9 should be post-tool (i>0 and i%3==0).
        assert [t.turn_index for t in tr.post_tool_turns] == [3, 6, 9]

    def test_mean_ttft_with_turns(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=_make_turns(10),
        )
        # Mean of 100..109 == 104.5
        assert tr.mean_ttft_ms == pytest.approx(104.5)

    def test_mean_ttft_empty(self) -> None:
        tr = TraceResult(config_name="baseline", trace_id="t1")
        assert tr.mean_ttft_ms == 0.0

    def test_mean_prefill_recomp_only_post_tool(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=_make_turns(10, post_tool_every=3),
        )
        # All post-tool turns have prefill_recomp_ms == 50.0
        assert tr.mean_prefill_recomp_ms == 50.0

    def test_mean_prefill_recomp_no_post_tool(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=[
                TurnMetrics(
                    turn_index=0,
                    ttft_ms=100,
                    tbt_ms_mean=10,
                    tbt_ms_p99=15,
                    wallclock_ms=200,
                )
            ],
        )
        assert tr.mean_prefill_recomp_ms == 0.0

    def test_p99_tbt_takes_max(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=_make_turns(10),
        )
        # p99 values are 15..19; max should be 19.
        assert tr.p99_tbt_ms == 19.0

    def test_xfer_bandwidth_zero_when_no_transfer(self) -> None:
        tr = TraceResult(config_name="baseline", trace_id="t1")
        assert tr.cpu_gpu_xfer_bandwidth_mb_s == 0.0

    def test_xfer_bandwidth_computation(self) -> None:
        # 100 MB transferred in 1000 ms == 100 MB/s
        tr = TraceResult(
            config_name="retention",
            trace_id="t1",
            cpu_gpu_xfer_bytes=100_000_000,
            cpu_gpu_xfer_ms=1000.0,
        )
        assert tr.cpu_gpu_xfer_bandwidth_mb_s == pytest.approx(100.0, rel=1e-3)

    def test_emit_json_round_trip(self, tmp_path) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="50turn",
            turns=_make_turns(5),
            peak_vram_mb=38000.0,
            preemption_count=2,
            metadata={"git_sha": "abc1234"},
        )
        path = tmp_path / "trace.json"
        tr.emit_json(path)

        with open(path) as f:
            blob = json.load(f)
        assert blob["config_name"] == "baseline"
        assert blob["preemption_count"] == 2
        assert len(blob["turns"]) == 5
        assert blob["metadata"]["git_sha"] == "abc1234"

        loaded = TraceResult.from_json(path)
        assert loaded.config_name == tr.config_name
        assert loaded.peak_vram_mb == tr.peak_vram_mb
        assert len(loaded.turns) == 5
        assert loaded.turns[0].turn_index == 0


# ---------------------------------------------------------------------------
# RunSummary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def test_aggregates_single_trace(self) -> None:
        tr = TraceResult(
            config_name="baseline",
            trace_id="t1",
            turns=_make_turns(10),
            peak_vram_mb=38000.0,
            preemption_count=2,
        )
        s = RunSummary.from_traces([tr])
        assert s.config_name == "baseline"
        assert s.num_traces == 1
        assert s.mean_ttft_ms == pytest.approx(104.5)
        assert s.total_preemptions == 2
        assert s.mean_peak_vram_mb == 38000.0

    def test_aggregates_multiple_traces(self) -> None:
        traces = [
            TraceResult(
                config_name="retention",
                trace_id=f"t{i}",
                turns=_make_turns(5),
                peak_vram_mb=30000.0 + i * 100,
                preemption_count=i,
            )
            for i in range(3)
        ]
        s = RunSummary.from_traces(traces)
        assert s.num_traces == 3
        assert s.total_preemptions == 0 + 1 + 2
        # Mean of 30000, 30100, 30200 == 30100
        assert s.mean_peak_vram_mb == pytest.approx(30100.0)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            RunSummary.from_traces([])

    def test_rejects_mixed_configs(self) -> None:
        traces = [
            TraceResult(config_name="a", trace_id="t1", turns=_make_turns(3)),
            TraceResult(config_name="b", trace_id="t2", turns=_make_turns(3)),
        ]
        with pytest.raises(ValueError, match="uniform config_name"):
            RunSummary.from_traces(traces)

    def test_handles_no_accuracy(self) -> None:
        tr = TraceResult(config_name="x", trace_id="t1", turns=_make_turns(3))
        s = RunSummary.from_traces([tr])
        assert s.mean_task_accuracy is None

    def test_aggregates_partial_accuracy(self) -> None:
        traces = [
            TraceResult(config_name="x", trace_id="t1", turns=_make_turns(3), task_accuracy=0.9),
            TraceResult(config_name="x", trace_id="t2", turns=_make_turns(3)),
            TraceResult(config_name="x", trace_id="t3", turns=_make_turns(3), task_accuracy=0.8),
        ]
        s = RunSummary.from_traces(traces)
        # Only two traces reported accuracy; mean should average those two.
        assert s.mean_task_accuracy == pytest.approx(0.85)
