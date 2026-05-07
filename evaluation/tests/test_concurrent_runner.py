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


class TestPromptUniqueness:
    """Regression: prompts for different concurrent agents must NOT be identical.

    If they are, vLLM's prefix cache hits at 100% across all agents and there
    is no cross-agent contention for retention to mitigate — masking any
    real difference between PC-only and retention configs.
    """

    class _CapturingEngine:
        """Mock engine that records every prompt it receives."""
        llm_engine = None  # not "real" — drives mock generate path

        def __init__(self) -> None:
            self.prompts_seen: list[str] = []
            self._counter = 0

        def generate(self, prompts, sampling_params=None, **kwargs):
            from evaluation.engine_adapter import (
                _MockCompletionOutput,
                _MockMetrics,
                _MockRequestOutput,
            )
            if isinstance(prompts, str):
                prompts = [prompts]
            results = []
            for prompt in prompts:
                self.prompts_seen.append(prompt)
                arrival = time.monotonic()
                results.append(_MockRequestOutput(
                    request_id=f"capt-{self._counter}",
                    prompt=prompt,
                    metrics=_MockMetrics(
                        arrival_time=arrival,
                        first_scheduled_time=arrival,
                        first_token_time=arrival + 0.05,
                        last_token_time=arrival + 0.10,
                    ),
                    outputs=[_MockCompletionOutput(
                        text="x", token_ids=[1, 2, 3]
                    )],
                ))
                self._counter += 1
            return results

    def test_concurrent_agents_get_unique_prompts(self) -> None:
        """Two replicated traces with different trace_ids must produce
        DIFFERENT prompts at the same turn index — proves trace_id is in
        the prompt body and PC won't share blocks across agents."""
        engine = self._CapturingEngine()
        specs = [
            fixture_trace(trace_id=f"agent_{i}", num_turns=2, tool_every=999)
            for i in range(2)
        ]
        run_concurrent(engine, specs, cfg={"name": "uniq"}, concurrency=2)

        # 2 traces × 2 turns each = 4 prompts captured.
        assert len(engine.prompts_seen) == 4

        # Group by turn_index — agents at the same turn must have DIFFERENT prompts.
        turn0 = [p for p in engine.prompts_seen if "turn=0" in p]
        assert len(turn0) == 2
        assert turn0[0] != turn0[1], (
            "Two agents at turn 0 produced identical prompts — "
            "trace_id is not being injected into the prompt body. "
            "vLLM's prefix cache will hit at 100% across agents and "
            "retention will appear to add no value."
        )

        # And both should reference their respective program_ids.
        assert any("agent_0" in p for p in turn0)
        assert any("agent_1" in p for p in turn0)


# ---------------------------------------------------------------------------
# Filler+focal mode regressions
#
# These tests cover the Daksh-microbenchmark code path:
#
#   - role="focal" / role="filler" propagates from TraceSpec through
#     _TraceState into TraceResult.metadata.
#   - The runner exits as soon as all focal traces complete (drain phase),
#     even if filler traces have many turns left.
#   - Per-trace TraceResult records role="focal" / "filler" so downstream
#     analysis can stratify focal-TTFT vs filler-TTFT.
#   - Filler traces submit user_prompt requests continuously (no tool gap
#     idle time), creating eviction pressure during focal's tool gap.
# ---------------------------------------------------------------------------

class TestFillerFocalMode:
    def test_role_propagates_to_metadata(self) -> None:
        """role="focal" / "filler" must end up in TraceResult.metadata.role
        so analysis tools can split focal-TTFT from filler-TTFT."""
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=2,
            focal_num_user_prompts=2,
            focal_tool_latency_ms=10.0,  # 10 ms — kept tiny for unit test
            filler_num_turns=5,
            focal_body_tokens_per_turn=64,
            filler_body_tokens_per_turn=64,
        )
        # 1 focal + 2 fillers = 3 specs.
        assert len(specs) == 3
        assert specs[0].role == "focal"
        assert all(s.role == "filler" for s in specs[1:])

        out = run_concurrent(MockEngine(), specs, cfg={"name": "ff"}, concurrency=3)

        roles = [tr.metadata.get("role") for tr in out]
        assert sorted(roles) == ["filler", "filler", "focal"]

        focal_results = [tr for tr in out if tr.metadata.get("role") == "focal"]
        filler_results = [tr for tr in out if tr.metadata.get("role") == "filler"]
        assert len(focal_results) == 1
        assert len(filler_results) == 2

    def test_focal_completion_terminates_run_early(self) -> None:
        """Once focal traces are done, fillers must NOT keep running their
        full turn count — the runner should drain in-flight and exit. With
        focal_num_user_prompts=1 and filler_num_turns=200, the focal trace
        takes ~1 turn worth of work; if termination logic is broken, the run
        would process 2 × 200 = 400 filler turns instead."""
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=2,
            focal_num_user_prompts=1,
            focal_tool_latency_ms=0.0,
            filler_num_turns=200,
            focal_body_tokens_per_turn=64,
            filler_body_tokens_per_turn=64,
        )
        out = run_concurrent(MockEngine(), specs, cfg={"name": "ff"}, concurrency=3)

        focal = next(tr for tr in out if tr.metadata.get("role") == "focal")
        fillers = [tr for tr in out if tr.metadata.get("role") == "filler"]

        # Focal completed all its user_prompts (just 1).
        assert len(focal.turns) == 1

        # Each filler ran SOME turns (not zero, not all 200). Mock engine is
        # synchronous so fillers and focal interleave on the for-loop pass.
        # In practice we expect the fillers to have run roughly 1–10 turns
        # before the focal completed and triggered drain.
        for f in fillers:
            assert 0 < len(f.turns) < 200, (
                f"Filler {f.trace_id} ran {len(f.turns)} turns — "
                "expected drain to terminate well before all 200 turns."
            )

    def test_focal_only_workload_unchanged_behaviour(self) -> None:
        """Pure-focal workloads (no fillers) must behave exactly like the
        pre-filler-aware runner: all states must finish, all turn counts
        must hit their full spec."""
        # All-focal: equivalent to legacy "replicated traces".
        specs = [
            fixture_trace(trace_id=f"f{i}", num_turns=3, tool_every=999, role="focal")
            for i in range(2)
        ]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "all-focal"}, concurrency=2)
        assert len(out) == 2
        for tr in out:
            assert len(tr.turns) == 3
            assert tr.metadata["role"] == "focal"

    def test_no_role_specified_defaults_to_focal(self) -> None:
        """Existing callers that don't pass role= must keep working — the
        default is "focal" so the legacy "loop until all done" path still
        applies."""
        # No role keyword — relies on TraceSpec default.
        specs = [fixture_trace(trace_id=f"t{i}", num_turns=2, tool_every=999) for i in range(2)]
        for s in specs:
            assert s.role == "focal"
        out = run_concurrent(MockEngine(), specs, cfg={"name": "default"}, concurrency=2)
        assert all(tr.metadata["role"] == "focal" for tr in out)

    def test_filler_user_prompts_have_no_tool_gaps(self) -> None:
        """Filler traces must consist purely of user_prompt turns with no
        tool_call/tool_return — they're meant to submit continuously to
        create cache pressure, not to spend time idle in tool gaps."""
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=1,
            focal_num_user_prompts=2,
            focal_tool_latency_ms=10.0,
            filler_num_turns=5,
            focal_body_tokens_per_turn=64,
            filler_body_tokens_per_turn=64,
        )
        focal_spec = specs[0]
        filler_spec = specs[1]

        assert all(t.kind == "user_prompt" for t in filler_spec.turns), (
            "Filler trace contains non-user_prompt turns — defeats the purpose "
            "of continuous-submission pressure."
        )
        # Focal should have at least one tool gap (between two user_prompts).
        focal_tool_calls = [t for t in focal_spec.turns if t.kind == "tool_call"]
        assert len(focal_tool_calls) == 1  # 2 user_prompts → 1 gap between

    def test_focal_tool_latency_passes_through(self) -> None:
        """Focal trace's tool_latency_ms in the spec should match what the
        builder was given — it's the parameter retention's TTL must cover."""
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=0,
            focal_num_user_prompts=2,
            focal_tool_latency_ms=12345.0,
        )
        focal = specs[0]
        tool_calls = [t for t in focal.turns if t.kind == "tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_latency_ms == 12345.0


class TestStartOffsetSemantics:
    """Regression: TraceSpec.start_offset_ms must defer the first request.

    Used by the staggered / burst workload classes to model arrivals over
    time. Without this, all agents would start at t=0 regardless of their
    nominal arrival schedule, collapsing back to lockstep.
    """

    def test_zero_offset_starts_immediately(self) -> None:
        spec = fixture_trace(num_turns=2, tool_every=999)  # default offset=0
        assert spec.start_offset_ms == 0.0
        out = run_concurrent(MockEngine(), [spec], cfg={"name": "imm"}, concurrency=1)
        assert len(out[0].turns) == 2

    def test_nonzero_offset_delays_first_metric(self) -> None:
        """A 100ms start_offset_ms should mean wallclock_ms includes that
        delay (mock engine timestamps via time.monotonic so the offset is
        observable in the metrics)."""
        spec = TraceSpec(
            trace_id="delayed",
            model="mock",
            prompt_tokens=64,
            tool_latency_dist="zero",
            turns=[TraceTurn(turn_index=0, kind="user_prompt", tokens=32)],
            start_offset_ms=100.0,
        )
        # Reference: a non-delayed spec for comparison.
        ref = TraceSpec(
            trace_id="immediate",
            model="mock",
            prompt_tokens=64,
            tool_latency_dist="zero",
            turns=[TraceTurn(turn_index=0, kind="user_prompt", tokens=32)],
        )
        # MockEngine TTFT is deterministic; our metric is "did the runner
        # actually wait?". Use time.monotonic before/after.
        t0 = time.monotonic()
        run_concurrent(MockEngine(), [spec, ref], cfg={"name": "delay"}, concurrency=2)
        elapsed = time.monotonic() - t0
        # 100 ms offset enforced; allow generous slack for CI variance.
        assert elapsed >= 0.080, (
            f"Run finished in {elapsed*1000:.1f}ms — offset of 100ms not honoured."
        )

    def test_offsets_do_not_break_focal_termination(self) -> None:
        """Combine start_offset with a non-filler workload — runner should
        still wait for every state to finish (no premature exit)."""
        specs = [
            TraceSpec(
                trace_id=f"agent-{i}",
                model="mock",
                prompt_tokens=64,
                tool_latency_dist="zero",
                turns=[TraceTurn(turn_index=0, kind="user_prompt", tokens=32),
                       TraceTurn(turn_index=1, kind="user_prompt", tokens=32)],
                start_offset_ms=10.0 * i,
            )
            for i in range(3)
        ]
        out = run_concurrent(MockEngine(), specs, cfg={"name": "off"}, concurrency=3)
        assert len(out) == 3
        # Each agent fully completed despite different offsets.
        assert all(len(tr.turns) == 2 for tr in out)


class TestFillerFocalBuilder:
    """Direct unit tests for trace_loader.filler_focal_workload."""

    def test_zero_fillers_gives_only_focal(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(num_fillers=0, focal_num_user_prompts=3)
        assert len(specs) == 1
        assert specs[0].role == "focal"

    def test_negative_fillers_raises(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        with pytest.raises(ValueError, match="num_fillers"):
            filler_focal_workload(num_fillers=-1)

    def test_zero_focal_user_prompts_raises(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        with pytest.raises(ValueError, match="focal_num_user_prompts"):
            filler_focal_workload(num_fillers=2, focal_num_user_prompts=0)

    def test_focal_has_n_user_prompts_and_nminus1_tool_gaps(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(num_fillers=0, focal_num_user_prompts=4)
        focal = specs[0]
        n_user = sum(1 for t in focal.turns if t.kind == "user_prompt")
        n_tc = sum(1 for t in focal.turns if t.kind == "tool_call")
        n_tr = sum(1 for t in focal.turns if t.kind == "tool_return")
        assert n_user == 4
        assert n_tc == 3
        assert n_tr == 3

    def test_filler_ids_are_unique(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(num_fillers=5, focal_num_user_prompts=2)
        filler_ids = [s.trace_id for s in specs if s.role == "filler"]
        assert len(set(filler_ids)) == len(filler_ids)
        assert filler_ids == ["filler-0", "filler-1", "filler-2", "filler-3", "filler-4"]
