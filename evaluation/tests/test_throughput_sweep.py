"""Tests for scripts/run_throughput_sweep.py — Path B (saturation curve)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


class TestFindSaturation:
    """Pure-function unit tests for the saturation-rate finder."""

    def test_monotonic_curve_finds_threshold(self) -> None:
        import run_throughput_sweep
        # p95 grows monotonically; SLO=300ms; saturation should be 4 (last
        # rate where p95 < 300).
        points = [
            {"rate": 1, "p95_ms": 100},
            {"rate": 2, "p95_ms": 150},
            {"rate": 4, "p95_ms": 250},
            {"rate": 8, "p95_ms": 400},
            {"rate": 16, "p95_ms": 800},
        ]
        sat = run_throughput_sweep.find_saturation(points, slo_ms=300.0)
        assert sat == 4

    def test_all_below_slo_returns_max_rate(self) -> None:
        import run_throughput_sweep
        points = [
            {"rate": 1, "p95_ms": 50},
            {"rate": 2, "p95_ms": 60},
            {"rate": 4, "p95_ms": 70},
        ]
        # All under SLO=300; return the highest rate (most-stressed point that still passed)
        sat = run_throughput_sweep.find_saturation(points, slo_ms=300.0)
        assert sat == 4

    def test_all_above_slo_returns_none(self) -> None:
        import run_throughput_sweep
        points = [
            {"rate": 1, "p95_ms": 500},
            {"rate": 2, "p95_ms": 600},
        ]
        sat = run_throughput_sweep.find_saturation(points, slo_ms=300.0)
        assert sat is None

    def test_missing_p95_skipped(self) -> None:
        """A run that crashed (missing p95_ms) shouldn't be treated as
        passing the SLO."""
        import run_throughput_sweep
        points = [
            {"rate": 1, "p95_ms": 100},
            {"rate": 2, "p95_ms": None, "missing": True},
            {"rate": 4, "p95_ms": 200},
            {"rate": 8, "p95_ms": 800},
        ]
        sat = run_throughput_sweep.find_saturation(points, slo_ms=300.0)
        assert sat == 4  # the missing rate=2 doesn't count, but 4 is still the saturation

    def test_empty_returns_none(self) -> None:
        import run_throughput_sweep
        assert run_throughput_sweep.find_saturation([], slo_ms=300.0) is None

    def test_uses_p99_metric_if_specified(self) -> None:
        import run_throughput_sweep
        points = [
            {"rate": 1, "p95_ms": 100, "p99_ms": 200},
            {"rate": 2, "p95_ms": 150, "p99_ms": 350},  # p99 > 300
            {"rate": 4, "p95_ms": 200, "p99_ms": 600},
        ]
        # SLO=300 on p99: only rate=1 passes
        sat = run_throughput_sweep.find_saturation(points, slo_ms=300.0,
                                                   metric="p99_ms")
        assert sat == 1


class TestCollectCurve:
    """Test the disk-walking aggregator."""

    def _make_summary(self, dir_path: Path, p95_ms: float, p99_ms: float = 0,
                     mean_ms: float = 0, n: int = 10) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        summary = {
            "config_name": dir_path.parent.name,
            "workload": {
                "post_tool_ttft": {
                    "mean_ms": mean_ms or p95_ms * 0.7,
                    "p95_ms": p95_ms,
                    "p99_ms": p99_ms or p95_ms * 1.05,
                    "n": n,
                },
                "all_ttft": {"mean_ms": p95_ms * 0.5, "p95_ms": p95_ms * 0.9},
            },
        }
        with open(dir_path / "summary.json", "w") as f:
            json.dump(summary, f)

    def test_collect_walks_all_rates(self, tmp_path: Path) -> None:
        import run_throughput_sweep
        configs = ["baseline_filler", "retention_filler"]
        rates = [1.0, 2.0, 4.0]
        for cfg in configs:
            for rate in rates:
                d = tmp_path / cfg / f"rate-{rate:g}"
                self._make_summary(d, p95_ms=100 * rate)
        curve = run_throughput_sweep.collect_curve(tmp_path, configs, rates)
        assert set(curve.keys()) == set(configs)
        assert len(curve["baseline_filler"]) == 3
        assert curve["baseline_filler"][0]["p95_ms"] == 100.0
        assert curve["baseline_filler"][2]["p95_ms"] == 400.0

    def test_collect_handles_missing_summary(self, tmp_path: Path) -> None:
        import run_throughput_sweep
        configs = ["a"]
        rates = [1.0, 2.0]
        # Only create rate=1 summary
        self._make_summary(tmp_path / "a" / "rate-1", p95_ms=100)
        curve = run_throughput_sweep.collect_curve(tmp_path, configs, rates)
        assert curve["a"][0]["p95_ms"] == 100.0
        assert curve["a"][1]["p95_ms"] is None
        assert curve["a"][1].get("missing") is True


class TestParseArgs:
    """CLI argparse smoke tests."""

    def test_arrival_rates_default(self, monkeypatch) -> None:
        import run_throughput_sweep
        monkeypatch.setattr(sys, "argv", [
            "run_throughput_sweep", "--output", "out/"
        ])
        args = run_throughput_sweep.parse_args()
        assert args.arrival_rates == "1,2,4,8,16"
        assert args.slo_ms == 500.0

    def test_custom_rates_parse(self, monkeypatch) -> None:
        import run_throughput_sweep
        monkeypatch.setattr(sys, "argv", [
            "run_throughput_sweep", "--output", "out/",
            "--arrival-rates", "0.5,1,2,4,8,16,32",
        ])
        args = run_throughput_sweep.parse_args()
        rates = [float(x) for x in args.arrival_rates.split(",")]
        assert rates == [0.5, 1, 2, 4, 8, 16, 32]
