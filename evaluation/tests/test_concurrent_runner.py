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


# ---------------------------------------------------------------------------
# Regression: fake real-engine step() pattern
#
# This is the path that broke during the L4 concurrent matrix run: vLLM's
# step() returns intermediate (unfinished) outputs while it's mid-decode,
# and the runner was incorrectly treating "no finished outputs this step"
# as "stuck" → break, producing empty TurnMetrics arrays.
#
# The fake engine here exposes a `.llm_engine` attribute (so _is_real_engine
# returns True) and emulates the multi-step pattern where it takes N step()
# calls before output.finished is True for a given request.
# ---------------------------------------------------------------------------

class _FakeRealEngine:
    """Mimics enough of vllm.LLM to exercise the engine.llm_engine.add_request
    + engine.llm_engine.step() path in concurrent_runner.

    Each request takes ``steps_per_request`` step() invocations to finish.
    During those intermediate steps, step() returns either an empty list or
    an unfinished output (depending on the mode), which is exactly what
    triggered the production bug.
    """

    def __init__(self, steps_per_request: int = 3, ttft_ms: float = 50.0,
                 per_token_ms: float = 5.0, n_output_tokens: int = 4):
        self.llm_engine = self  # so concurrent_runner detects "real engine"
        self._pending = {}      # request_id → steps remaining
        self._params = {}       # request_id → (arrival_time, prompt)
        self.steps_per_request = steps_per_request
        self.ttft_ms = ttft_ms
        self.per_token_ms = per_token_ms
        self.n_output_tokens = n_output_tokens

    def add_request(self, request_id, prompt, sampling_params, **_kwargs):
        self._pending[request_id] = self.steps_per_request
        self._params[request_id] = (time.monotonic(), prompt)

    def step(self):
        from evaluation.engine_adapter import (
            _MockCompletionOutput,
            _MockMetrics,
            _MockRequestOutput,
        )
        outputs = []
        finished_ids = []
        for rid, remaining in list(self._pending.items()):
            self._pending[rid] = remaining - 1
            if self._pending[rid] <= 0:
                # Synthesize a finished output with realistic metrics
                arrival, prompt = self._params[rid]
                first_token = arrival + self.ttft_ms / 1000.0
                last_token = first_token + (self.n_output_tokens * self.per_token_ms / 1000.0)
                outputs.append(_MockRequestOutput(
                    request_id=rid,
                    prompt=prompt,
                    metrics=_MockMetrics(
                        arrival_time=arrival,
                        first_scheduled_time=arrival,
                        first_token_time=first_token,
                        last_token_time=last_token,
                        time_in_queue=0.0,
                        finished_time=last_token,
                    ),
                    outputs=[_MockCompletionOutput(
                        text="x " * self.n_output_tokens,
                        token_ids=list(range(self.n_output_tokens)),
                    )],
                    finished=True,
                ))
                finished_ids.append(rid)
        for rid in finished_ids:
            del self._pending[rid]
            del self._params[rid]
        # Intermediate steps return EMPTY list — this is the case the buggy
        # version mishandled (treated empty step output as "stuck").
        return outputs


class TestRealEnginePath:
    def test_multistep_finish_does_not_break_loop(self) -> None:
        """Reproduces the L4 production bug: real engine takes multiple step()
        calls per request. The buggy version exited the loop on the first
        empty step() and produced 0 turns. The fix should produce per-turn
        metrics for every user_prompt."""
        engine = _FakeRealEngine(steps_per_request=5)
        spec = fixture_trace(num_turns=4, tool_every=999)  # all user_prompt
        out = run_concurrent(engine, [spec], cfg={"name": "regress"}, concurrency=1)

        assert len(out) == 1
        assert len(out[0].turns) == 4, (
            f"Expected 4 turn metrics, got {len(out[0].turns)} — "
            "runner exited before vLLM step loop finished requests."
        )
        for tm in out[0].turns:
            assert tm.ttft_ms > 0, f"ttft_ms should be > 0 from real-engine path, got {tm.ttft_ms}"
            assert tm.ttft_ms != -1.0

    def test_concurrent_traces_through_multistep_engine(self) -> None:
        """Multiple traces concurrently with multi-step engine → all traces
        should produce full TurnMetrics arrays."""
        engine = _FakeRealEngine(steps_per_request=3, ttft_ms=80.0)
        specs = [
            fixture_trace(trace_id=f"agent{i}", num_turns=3, tool_every=999)
            for i in range(4)
        ]
        out = run_concurrent(engine, specs, cfg={"name": "regress"}, concurrency=4)

        assert len(out) == 4
        for tr in out:
            assert len(tr.turns) == 3
            for tm in tr.turns:
                assert tm.ttft_ms == pytest.approx(80.0, abs=5.0)
