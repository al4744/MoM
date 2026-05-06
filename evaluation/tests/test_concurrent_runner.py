"""Tests for evaluation/concurrent_runner.py — multi-trace concurrent driver."""
from __future__ import annotations

import time

import pytest

from evaluation.concurrent_runner import (
    _TraceState,
    _is_real_engine,
    run_concurrent,
)
from evaluation.engine_adapter import MockEngine
from evaluation.metrics import RunSummary, TraceResult
from evaluation.trace_loader import TraceSpec, TraceTurn, fixture_trace


# ---------------------------------------------------------------------------
# _TraceState
# ---------------------------------------------------------------------------

class TestTraceState:
    def test_finished_when_no_more_turns_and_no_pending(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        state = _TraceState(spec=spec, turn_idx=2)
        assert state.finished

    def test_not_finished_when_pending_request(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        state = _TraceState(
            spec=spec,
            turn_idx=2,
            pending_request_id="r-0",
        )
        assert not state.finished

    def test_not_finished_when_in_tool_gap(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        state = _TraceState(
            spec=spec,
            turn_idx=2,
            tool_gap_ends=time.monotonic() + 60.0,
        )
        assert not state.finished


# ---------------------------------------------------------------------------
# _is_real_engine detection
# ---------------------------------------------------------------------------

class TestIsRealEngine:
    def test_mock_engine_is_not_real(self) -> None:
        assert _is_real_engine(MockEngine()) is False

    def test_object_with_llm_engine_attr_is_real(self) -> None:
        class FakeLLM:
            llm_engine = object()
        assert _is_real_engine(FakeLLM()) is True

    def test_object_without_llm_engine_attr_is_not_real(self) -> None:
        class Bare:
            pass
        assert _is_real_engine(Bare()) is False


# ---------------------------------------------------------------------------
# run_concurrent — happy paths
# ---------------------------------------------------------------------------

class TestRunConcurrentBasics:
    def test_returns_one_traceresult_per_spec(self) -> None:
        specs = [
            fixture_trace(trace_id="a", num_turns=3, tool_every=999),
            fixture_trace(trace_id="b", num_turns=3, tool_every=999),
        ]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "test"}, concurrency=2)
        assert len(out) == 2
        assert {t.trace_id for t in out} == {"a", "b"}

    def test_each_trace_has_per_turn_metrics(self) -> None:
        specs = [
            fixture_trace(trace_id=f"t{i}", num_turns=4, tool_every=999) for i in range(3)
        ]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "test"}, concurrency=3)
        for tr in out:
            assert isinstance(tr, TraceResult)
            assert len(tr.turns) == 4
            for tm in tr.turns:
                assert tm.ttft_ms > 0
                assert tm.ttft_ms != -1.0  # not a sentinel

    def test_concurrency_metadata_recorded(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        out = run_concurrent(MockEngine(), [spec], cfg={"name": "test"}, concurrency=4)
        assert out[0].metadata["concurrency"] == 4
        assert out[0].metadata["num_concurrent_specs"] == 1

    def test_concurrency_one_works_like_sequential(self) -> None:
        specs = [
            fixture_trace(trace_id=f"t{i}", num_turns=3, tool_every=999) for i in range(2)
        ]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "test"}, concurrency=1)
        assert len(out) == 2
        assert all(len(tr.turns) == 3 for tr in out)


# ---------------------------------------------------------------------------
# run_concurrent — tool-call handling
# ---------------------------------------------------------------------------

class TestRunConcurrentToolHandling:
    def test_tool_call_turns_skipped_in_metrics(self) -> None:
        spec = fixture_trace(num_turns=9, tool_every=3)  # mix of u/tc/tr turns
        n_user = sum(1 for t in spec.turns if t.kind == "user_prompt")
        out = run_concurrent(MockEngine(), [spec], cfg={"name": "test"}, concurrency=1)
        assert len(out[0].turns) == n_user

    def test_post_tool_turns_marked(self) -> None:
        # Hand-craft a spec: u, tc, tr, u — second user_prompt is post-tool.
        spec = TraceSpec(
            trace_id="hand",
            model="mock",
            prompt_tokens=64,
            tool_latency_dist="zero",
            turns=[
                TraceTurn(turn_index=0, kind="user_prompt", tokens=32),
                TraceTurn(turn_index=1, kind="tool_call", tokens=8,
                          tool_name="search", tool_latency_ms=0.0),
                TraceTurn(turn_index=2, kind="tool_return", tokens=16,
                          tool_name="search"),
                TraceTurn(turn_index=3, kind="user_prompt", tokens=32),
            ],
        )
        out = run_concurrent(MockEngine(), [spec], cfg={"name": "test"}, concurrency=1)
        assert len(out[0].turns) == 2
        # First user_prompt: not post-tool.
        assert out[0].turns[0].prefill_recomp_ms is None
        # Second user_prompt at index 3 follows tool_return at index 2 → marked.
        assert out[0].turns[1].prefill_recomp_ms is not None


# ---------------------------------------------------------------------------
# run_concurrent — multi-trace summary aggregation
# ---------------------------------------------------------------------------

class TestRunConcurrentSummary:
    def test_runsummary_aggregates_finite_values(self) -> None:
        specs = [fixture_trace(trace_id=f"t{i}", num_turns=3, tool_every=999) for i in range(4)]
        out = run_concurrent(
            MockEngine(ttft_ms=120.0, per_token_ms=2.0),
            specs,
            cfg={"name": "concurrent-smoke"},
            concurrency=2,
        )
        summary = RunSummary.from_traces(out)
        assert summary.config_name == "concurrent-smoke"
        assert summary.num_traces == 4
        assert summary.mean_ttft_ms > 0
        assert summary.mean_ttft_ms != -1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestRunConcurrentEdgeCases:
    def test_zero_concurrency_raises(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            run_concurrent(MockEngine(), [spec], cfg={"name": "x"}, concurrency=0)

    def test_negative_concurrency_raises(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)
        with pytest.raises(ValueError, match="concurrency must be >= 1"):
            run_concurrent(MockEngine(), [spec], cfg={"name": "x"}, concurrency=-1)

    def test_more_concurrency_than_specs_works(self) -> None:
        # concurrency=10 but only 2 specs — should work fine.
        specs = [fixture_trace(trace_id=f"t{i}", num_turns=2, tool_every=999) for i in range(2)]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "test"}, concurrency=10)
        assert len(out) == 2
