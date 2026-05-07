"""Tests for evaluation/workloads.py — the 5-class workload battery.

Coverage:
  * Each builder produces the right number of agents with the right turn shape.
  * Validation errors fire for nonsense parameters (zero agents, etc.).
  * Class-specific invariants:
      - lockstep: all start_offset_ms == 0, identical tool latencies.
      - staggered: cumulative-sum monotonically increasing offsets, mean
                   inter-arrival ≈ 1/rate.
      - heterogeneous: tool latencies actually vary; arithmetic mean
                       roughly matches the requested mean.
      - burst: all offsets in [0, burst_duration_ms].
  * Each workload class round-trips through run_concurrent (MockEngine)
    and produces metric arrays of the expected shape.
"""
from __future__ import annotations

import statistics

import pytest

from evaluation.concurrent_runner import run_concurrent
from evaluation.engine_adapter import MockEngine
from evaluation.trace_loader import TraceSpec
from evaluation.workloads import (
    burst_workload,
    filler_focal_workload,
    heterogeneous_workload,
    lockstep_workload,
    staggered_workload,
)


# ---------------------------------------------------------------------------
# W1 — lockstep
# ---------------------------------------------------------------------------

class TestLockstepWorkload:
    def test_n_identical_agents(self) -> None:
        specs = lockstep_workload(num_agents=5, num_user_prompts=3)
        assert len(specs) == 5
        assert all(s.role == "focal" for s in specs)
        assert {s.trace_id for s in specs} == {f"lockstep-{i}" for i in range(5)}

    def test_all_start_at_zero(self) -> None:
        specs = lockstep_workload(num_agents=4, num_user_prompts=2)
        assert all(s.start_offset_ms == 0.0 for s in specs)

    def test_identical_tool_latency_across_agents(self) -> None:
        specs = lockstep_workload(num_agents=3, tool_latency_ms=1234.0)
        for s in specs:
            tool_calls = [t for t in s.turns if t.kind == "tool_call"]
            assert all(tc.tool_latency_ms == 1234.0 for tc in tool_calls)

    def test_zero_agents_raises(self) -> None:
        with pytest.raises(ValueError, match="num_agents"):
            lockstep_workload(num_agents=0, num_user_prompts=2)

    def test_zero_user_prompts_raises(self) -> None:
        with pytest.raises(ValueError, match="num_user_prompts"):
            lockstep_workload(num_agents=2, num_user_prompts=0)

    def test_runs_to_completion_via_runner(self) -> None:
        specs = lockstep_workload(num_agents=3, num_user_prompts=2,
                                  tool_latency_ms=10.0,
                                  body_tokens_per_turn=64)
        out = run_concurrent(MockEngine(), specs, cfg={"name": "ls"}, concurrency=3)
        assert len(out) == 3
        for tr in out:
            assert len(tr.turns) == 2
            assert tr.metadata["role"] == "focal"


# ---------------------------------------------------------------------------
# W3 — staggered
# ---------------------------------------------------------------------------

class TestStaggeredWorkload:
    def test_n_agents_with_increasing_offsets(self) -> None:
        specs = staggered_workload(
            num_agents=5,
            arrival_rate_per_sec=1.0,
            num_user_prompts=2,
            seed=42,
        )
        assert len(specs) == 5
        offsets = [s.start_offset_ms for s in specs]
        # First agent starts at 0; subsequent are cumulative sums of positive
        # exponential samples → strictly increasing.
        assert offsets[0] == 0.0
        for prev, cur in zip(offsets, offsets[1:]):
            assert cur > prev

    def test_role_focal(self) -> None:
        specs = staggered_workload(num_agents=3, arrival_rate_per_sec=2.0)
        assert all(s.role == "focal" for s in specs)

    def test_mean_interarrival_matches_rate(self) -> None:
        # 1000 agents @ 1.0/s should give mean inter-arrival ≈ 1.0s.
        specs = staggered_workload(
            num_agents=1000,
            arrival_rate_per_sec=1.0,
            num_user_prompts=1,
            seed=42,
        )
        offsets_s = [s.start_offset_ms / 1000.0 for s in specs]
        gaps = [b - a for a, b in zip(offsets_s, offsets_s[1:])]
        # Loose bound — exponential mean for n=999 is well within ±15%.
        mean_gap = statistics.mean(gaps)
        assert 0.85 < mean_gap < 1.15, (
            f"mean inter-arrival {mean_gap}s deviates >15% from expected 1.0s"
        )

    def test_zero_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="arrival_rate_per_sec"):
            staggered_workload(num_agents=2, arrival_rate_per_sec=0.0)

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="arrival_rate_per_sec"):
            staggered_workload(num_agents=2, arrival_rate_per_sec=-1.0)

    def test_seed_determinism(self) -> None:
        s1 = staggered_workload(num_agents=5, arrival_rate_per_sec=1.0, seed=42)
        s2 = staggered_workload(num_agents=5, arrival_rate_per_sec=1.0, seed=42)
        assert [s.start_offset_ms for s in s1] == [s.start_offset_ms for s in s2]

    def test_runs_to_completion_via_runner(self) -> None:
        # Tiny offsets to keep test fast (rate=100/s → average 10ms gaps).
        specs = staggered_workload(
            num_agents=3,
            arrival_rate_per_sec=100.0,
            num_user_prompts=2,
            tool_latency_ms=0.0,
            body_tokens_per_turn=64,
            seed=0,
        )
        out = run_concurrent(MockEngine(), specs, cfg={"name": "stag"}, concurrency=3)
        assert len(out) == 3
        # Every agent ran its full turn count.
        assert all(len(tr.turns) == 2 for tr in out)


# ---------------------------------------------------------------------------
# W4 — heterogeneous
# ---------------------------------------------------------------------------

class TestHeterogeneousWorkload:
    def test_n_agents_all_start_zero(self) -> None:
        specs = heterogeneous_workload(num_agents=4, num_user_prompts=2)
        assert len(specs) == 4
        assert all(s.start_offset_ms == 0.0 for s in specs)

    def test_tool_latencies_actually_vary(self) -> None:
        specs = heterogeneous_workload(
            num_agents=10,
            num_user_prompts=2,
            tool_latency_log_mean_ms=1500.0,
            tool_latency_log_sigma=0.7,
            seed=42,
        )
        # Each agent has 1 sampled latency; with sigma=0.7 we expect spread.
        per_agent_latencies = []
        for s in specs:
            tcs = [t for t in s.turns if t.kind == "tool_call"]
            assert tcs  # has at least one tool call
            per_agent_latencies.append(tcs[0].tool_latency_ms)
        # Distinct values across 10 agents (probability of collision ≈ 0).
        assert len(set(per_agent_latencies)) >= 8, (
            f"Expected spread but saw {len(set(per_agent_latencies))} "
            f"distinct values: {sorted(per_agent_latencies)}"
        )
        # Stdev should be non-trivial.
        assert statistics.stdev(per_agent_latencies) > 100.0

    def test_arithmetic_mean_matches_target(self) -> None:
        # 1000 agents → empirical mean ≈ requested log-normal mean.
        specs = heterogeneous_workload(
            num_agents=1000,
            num_user_prompts=2,
            tool_latency_log_mean_ms=1500.0,
            tool_latency_log_sigma=0.5,
            seed=42,
        )
        latencies = [
            [t.tool_latency_ms for t in s.turns if t.kind == "tool_call"][0]
            for s in specs
        ]
        empirical_mean = statistics.mean(latencies)
        # Loose: ±15% of target.
        assert 1500.0 * 0.85 < empirical_mean < 1500.0 * 1.15, (
            f"Empirical mean {empirical_mean:.1f} ms drifted from target 1500 ms"
        )

    def test_zero_log_sigma_gives_constant_latency(self) -> None:
        specs = heterogeneous_workload(
            num_agents=5,
            num_user_prompts=2,
            tool_latency_log_mean_ms=1000.0,
            tool_latency_log_sigma=0.0,
            seed=42,
        )
        latencies = [
            [t.tool_latency_ms for t in s.turns if t.kind == "tool_call"][0]
            for s in specs
        ]
        # All identical (or extremely close due to FP).
        assert max(latencies) - min(latencies) < 1e-6

    def test_negative_mean_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_latency_log_mean_ms"):
            heterogeneous_workload(num_agents=2, tool_latency_log_mean_ms=-1.0)

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_latency_log_sigma"):
            heterogeneous_workload(num_agents=2, tool_latency_log_sigma=-0.1)

    def test_runs_to_completion_via_runner(self) -> None:
        specs = heterogeneous_workload(
            num_agents=3,
            num_user_prompts=2,
            tool_latency_log_mean_ms=10.0,  # 10 ms, fast for unit test
            tool_latency_log_sigma=0.3,
            body_tokens_per_turn=64,
            seed=0,
        )
        out = run_concurrent(MockEngine(), specs, cfg={"name": "het"}, concurrency=3)
        assert len(out) == 3
        assert all(len(tr.turns) == 2 for tr in out)


# ---------------------------------------------------------------------------
# W5 — burst
# ---------------------------------------------------------------------------

class TestBurstWorkload:
    def test_n_agents_offsets_in_window(self) -> None:
        specs = burst_workload(
            num_agents=10,
            burst_duration_ms=500.0,
            num_user_prompts=2,
            seed=42,
        )
        assert len(specs) == 10
        offsets = [s.start_offset_ms for s in specs]
        assert all(0.0 <= o <= 500.0 for o in offsets)

    def test_role_focal(self) -> None:
        specs = burst_workload(num_agents=3, burst_duration_ms=200.0)
        assert all(s.role == "focal" for s in specs)

    def test_offsets_sorted(self) -> None:
        specs = burst_workload(num_agents=8, burst_duration_ms=1000.0, seed=42)
        offsets = [s.start_offset_ms for s in specs]
        assert offsets == sorted(offsets)

    def test_zero_duration_is_lockstep_equivalent(self) -> None:
        # burst_duration_ms=0 → all offsets == 0 (lockstep semantics).
        specs = burst_workload(num_agents=5, burst_duration_ms=0.0, seed=42)
        assert all(s.start_offset_ms == 0.0 for s in specs)

    def test_negative_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="burst_duration_ms"):
            burst_workload(num_agents=2, burst_duration_ms=-1.0)

    def test_single_agent_starts_at_zero(self) -> None:
        specs = burst_workload(num_agents=1, burst_duration_ms=500.0)
        assert specs[0].start_offset_ms == 0.0

    def test_runs_to_completion_via_runner(self) -> None:
        specs = burst_workload(
            num_agents=3,
            burst_duration_ms=20.0,  # tight window, small for unit test
            num_user_prompts=2,
            tool_latency_ms=0.0,
            body_tokens_per_turn=64,
            seed=0,
        )
        out = run_concurrent(MockEngine(), specs, cfg={"name": "burst"},
                             concurrency=3)
        assert len(out) == 3
        assert all(len(tr.turns) == 2 for tr in out)


# ---------------------------------------------------------------------------
# Re-export sanity — filler_focal_workload accessible via workloads module.
# ---------------------------------------------------------------------------

class TestFillerFocalReexport:
    def test_filler_focal_imported_from_workloads(self) -> None:
        specs = filler_focal_workload(num_fillers=2, focal_num_user_prompts=2)
        assert len(specs) == 3
        assert specs[0].role == "focal"


# ---------------------------------------------------------------------------
# Cross-class smoke: every workload class should produce role propagation
# all the way through to TraceResult.metadata via the concurrent runner.
# ---------------------------------------------------------------------------

class TestCrossClassRoleMetadata:
    @pytest.mark.parametrize("builder, kwargs", [
        (lockstep_workload, {"num_agents": 2, "num_user_prompts": 2,
                             "tool_latency_ms": 0.0, "body_tokens_per_turn": 64}),
        (staggered_workload, {"num_agents": 2, "arrival_rate_per_sec": 100.0,
                              "num_user_prompts": 2, "tool_latency_ms": 0.0,
                              "body_tokens_per_turn": 64, "seed": 0}),
        (heterogeneous_workload, {"num_agents": 2, "num_user_prompts": 2,
                                  "tool_latency_log_mean_ms": 5.0,
                                  "tool_latency_log_sigma": 0.2,
                                  "body_tokens_per_turn": 64, "seed": 0}),
        (burst_workload, {"num_agents": 2, "burst_duration_ms": 5.0,
                          "num_user_prompts": 2, "tool_latency_ms": 0.0,
                          "body_tokens_per_turn": 64, "seed": 0}),
    ])
    def test_focal_role_propagates(self, builder, kwargs) -> None:
        specs = builder(**kwargs)
        out = run_concurrent(MockEngine(), specs, cfg={"name": "cls"},
                             concurrency=2)
        assert all(tr.metadata["role"] == "focal" for tr in out)
