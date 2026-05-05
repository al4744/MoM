"""Tests for evaluation/comparison_table.py."""
from __future__ import annotations

import json

import pytest

from evaluation import comparison_table


def _summary(name: str, ttft: float, prt: float = 0.0, vram: float = 30000.0,
             prmp: int = 0, acc: float | None = None) -> dict:
    return {
        "config_name": name,
        "num_traces": 1,
        "mean_ttft_ms": ttft,
        "mean_prefill_recomp_ms": prt,
        "p99_tbt_ms": 15.0,
        "mean_peak_vram_mb": vram,
        "mean_quantized_kv_mb": None,
        "mean_kv_compression_ratio": None,
        "total_preemptions": prmp,
        "mean_xfer_bandwidth_mb_s": 0.0,
        "mean_task_accuracy": acc,
    }


class TestEmitTable:
    def test_sorts_by_ttft_ascending(self) -> None:
        summaries = [
            _summary("baseline", 480.0),
            _summary("retention", 165.0),
            _summary("full-stack", 121.0),
        ]
        out = comparison_table.emit_table(summaries)
        # Find the row index of each config; full-stack should come first.
        idx_full = out.index("full-stack")
        idx_retention = out.index("retention")
        idx_baseline = out.index("baseline")
        assert idx_full < idx_retention < idx_baseline

    def test_handles_missing_accuracy(self) -> None:
        out = comparison_table.emit_table([_summary("x", 100.0, acc=None)])
        # Em-dash sentinel for missing fields.
        assert "—" in out

    def test_renders_accuracy_with_three_decimals(self) -> None:
        out = comparison_table.emit_table([_summary("x", 100.0, acc=0.876)])
        assert "0.876" in out

    def test_includes_header_row(self) -> None:
        out = comparison_table.emit_table([_summary("x", 100.0)])
        assert "TTFT (ms)" in out
        assert "Prefill recomp (ms)" in out
        assert "VRAM (MB)" in out
        assert "Quant KV (MB)" in out


class TestLoadSummaries:
    def test_walks_results_root(self, tmp_path) -> None:
        for name, ttft in [("a", 100.0), ("b", 200.0), ("c", 50.0)]:
            d = tmp_path / name
            d.mkdir()
            with open(d / "summary.json", "w") as f:
                json.dump(_summary(name, ttft), f)

        loaded = comparison_table.load_summaries(tmp_path)
        assert len(loaded) == 3
        names = {s["config_name"] for s in loaded}
        assert names == {"a", "b", "c"}

    def test_empty_root_returns_empty(self, tmp_path) -> None:
        assert comparison_table.load_summaries(tmp_path) == []


class TestFmt:
    def test_none_renders_as_emdash(self) -> None:
        assert comparison_table.fmt(None) == "—"

    def test_int_renders_without_precision(self) -> None:
        assert comparison_table.fmt(42) == "42"

    def test_float_uses_default_precision(self) -> None:
        assert comparison_table.fmt(3.14159) == "3.1"

    def test_float_respects_precision_arg(self) -> None:
        assert comparison_table.fmt(3.14159, prec=3) == "3.142"
