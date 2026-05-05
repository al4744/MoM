"""Tests for evaluation/comparison_table.py --ablate mode."""
from __future__ import annotations

import pytest

from evaluation import comparison_table
from evaluation.comparison_table import (
    emit_ablate_table,
    fmt_delta,
    fmt_speedup,
)


def _summary(name: str, ttft: float, prt: float, vram: float = 30000.0,
             prmp: int = 0, acc: float | None = None,
             xfer: float = 0.0) -> dict:
    return {
        "config_name": name,
        "num_traces": 1,
        "mean_ttft_ms": ttft,
        "mean_prefill_recomp_ms": prt,
        "p99_tbt_ms": 15.0,
        "mean_peak_vram_mb": vram,
        "total_preemptions": prmp,
        "mean_xfer_bandwidth_mb_s": xfer,
        "mean_task_accuracy": acc,
    }


class TestFmtDelta:
    def test_positive_delta(self) -> None:
        assert fmt_delta(100.0, 150.0) == "+50.0"

    def test_negative_delta(self) -> None:
        assert fmt_delta(150.0, 100.0) == "-50.0"

    def test_int_delta(self) -> None:
        assert fmt_delta(10, 3) == "-7"

    def test_handles_none(self) -> None:
        assert fmt_delta(None, 5.0) == "—"
        assert fmt_delta(5.0, None) == "—"


class TestFmtSpeedup:
    def test_lower_is_better_improvement(self) -> None:
        # TTFT went from 480 to 165 → 2.91x speedup.
        assert fmt_speedup(480.0, 165.0, lower_is_better=True) == "2.91x"

    def test_lower_is_better_regression(self) -> None:
        # TTFT went from 100 to 200 → 0.50x (slowdown).
        assert fmt_speedup(100.0, 200.0, lower_is_better=True) == "0.50x"

    def test_higher_is_better_improvement(self) -> None:
        # Accuracy went from 0.5 to 0.9 → 1.80x improvement.
        assert fmt_speedup(0.5, 0.9, lower_is_better=False) == "1.80x"

    def test_zero_baseline_higher_better(self) -> None:
        # XFER went from 0 (no offload) to 100 MB/s. Speedup is undefined.
        assert fmt_speedup(0.0, 100.0, lower_is_better=False) == "—"

    def test_zero_candidate_lower_better(self) -> None:
        # Candidate hit zero (e.g. zero preemptions). Speedup undefined.
        assert fmt_speedup(10.0, 0.0, lower_is_better=True) == "—"

    def test_handles_none(self) -> None:
        assert fmt_speedup(None, 100.0, lower_is_better=True) == "—"


class TestEmitAblateTable:
    def test_renders_baseline_vs_candidate_columns(self) -> None:
        baseline = _summary("baseline", ttft=480.0, prt=320.0)
        candidate = _summary("retention", ttft=165.0, prt=12.8)
        table = emit_ablate_table(baseline, candidate)
        # Header includes both config names.
        assert "baseline" in table
        assert "retention" in table
        # Header has Δ and speedup columns.
        assert "Δ" in table
        assert "speedup" in table

    def test_includes_all_six_primary_metrics(self) -> None:
        baseline = _summary("a", ttft=100.0, prt=50.0)
        candidate = _summary("b", ttft=50.0, prt=10.0)
        table = emit_ablate_table(baseline, candidate)
        assert "TTFT" in table
        assert "Prefill recomp" in table
        assert "TBT p99" in table
        assert "VRAM" in table
        assert "Preemptions" in table
        assert "XFER" in table
        assert "Task accuracy" in table

    def test_speedup_math_visible_in_output(self) -> None:
        baseline = _summary("a", ttft=480.0, prt=320.0)
        candidate = _summary("b", ttft=160.0, prt=10.0)
        table = emit_ablate_table(baseline, candidate)
        # 480 / 160 == 3.00x; 320 / 10 == 32.00x
        assert "3.00x" in table
        assert "32.00x" in table

    def test_handles_missing_accuracy_gracefully(self) -> None:
        baseline = _summary("a", ttft=100.0, prt=50.0, acc=None)
        candidate = _summary("b", ttft=80.0, prt=40.0, acc=None)
        table = emit_ablate_table(baseline, candidate)
        # Should not raise; em-dash should appear for missing-accuracy row.
        assert "Task accuracy" in table
        assert "—" in table
