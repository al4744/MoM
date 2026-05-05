"""Smoke tests for run_eval.run_trace() and run_eval.run_real() against MockEngine.

These exercise the full real-run code path on CPU with no vLLM dependency,
verifying that:

  1. Per-turn metrics are populated from RequestOutput-shaped objects.
  2. Tool-call turns invoke the latency simulator (we use 0ms to keep tests fast).
  3. Post-tool turns get prefill_recomp_ms set; non-post-tool turns don't.
  4. run_real() walks every TraceSpec and returns a list of TraceResults.
  5. The aggregated RunSummary computes finite, non-sentinel numbers.
"""
from __future__ import annotations

import pytest

from evaluation.engine_adapter import MockEngine
from evaluation.metrics import RunSummary, TraceResult, TurnMetrics
from evaluation.run_eval import run_real, run_trace
from evaluation.trace_loader import TraceSpec, TraceTurn, fixture_trace


# ---------------------------------------------------------------------------
# run_trace — single trace through MockEngine
# ---------------------------------------------------------------------------

class TestRunTrace:
    def test_returns_traceresult(self) -> None:
        engine = MockEngine()
        spec = fixture_trace(num_turns=5, tool_every=999)  # no tools
        out = run_trace(engine, spec, cfg={"name": "test"})
        assert isinstance(out, TraceResult)
        assert out.config_name == "test"
        assert out.trace_id == spec.trace_id

    def test_one_turnmetric_per_user_prompt(self) -> None:
        engine = MockEngine()
        spec = fixture_trace(num_turns=6, tool_every=999)  # all user_prompt turns
        out = run_trace(engine, spec, cfg={"name": "test"})
        # All 6 turns are user_prompt → 6 TurnMetrics.
        assert len(out.turns) == 6

    def test_skips_tool_call_and_tool_return_turns(self) -> None:
        engine = MockEngine()
        # fixture_trace inserts (tool_call + tool_return) at every tool_every-th turn.
        # With num_turns=9, tool_every=3 → user_prompt turns at indices 0,1,2,4,5,7,8
        # but the budget is capped at 9 total spec.turns (mix of all kinds).
        spec = fixture_trace(num_turns=9, tool_every=3)
        user_prompt_count = sum(1 for t in spec.turns if t.kind == "user_prompt")
        out = run_trace(engine, spec, cfg={"name": "test"})
        assert len(out.turns) == user_prompt_count

    def test_marks_post_tool_turns(self) -> None:
        # Build a hand-crafted spec: u, tc, tr, u, u — second user_prompt
        # immediately follows tool_return so it should have prefill_recomp_ms set.
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
                TraceTurn(turn_index=4, kind="user_prompt", tokens=32),
            ],
        )
        out = run_trace(MockEngine(), spec, cfg={"name": "test"})
        # Three user_prompt turns produced TurnMetrics.
        assert len(out.turns) == 3
        # First user_prompt: not post-tool (no preceding tool_return).
        assert out.turns[0].prefill_recomp_ms is None
        # Second user_prompt at index 3: immediately follows tool_return → marked.
        assert out.turns[1].prefill_recomp_ms is not None
        # Third user_prompt at index 4: previous turn was a user_prompt, not tool_return.
        assert out.turns[2].prefill_recomp_ms is None

    def test_ttft_is_finite_and_positive(self) -> None:
        engine = MockEngine(ttft_ms=80.0, per_token_ms=2.0)
        spec = fixture_trace(num_turns=3, tool_every=999)
        out = run_trace(engine, spec, cfg={"name": "test"})
        for tm in out.turns:
            assert tm.ttft_ms > 0
            # Sentinel guard: should not match dry-run sentinel.
            assert tm.ttft_ms != -1.0

    def test_ttft_matches_mock_configuration(self) -> None:
        engine = MockEngine(ttft_ms=120.0, per_token_ms=0.0)
        spec = fixture_trace(num_turns=2, tool_every=999)
        out = run_trace(engine, spec, cfg={"name": "test"})
        assert out.turns[0].ttft_ms == pytest.approx(120.0, abs=1.0)

    def test_metadata_populated(self) -> None:
        engine = MockEngine()
        spec = fixture_trace(num_turns=2, tool_every=999, model="mock-model")
        out = run_trace(engine, spec, cfg={"name": "test"})
        assert out.metadata["model"] == "mock-model"
        assert "tool_latency_dist" in out.metadata


# ---------------------------------------------------------------------------
# run_real — multi-trace dispatch with injected engine
# ---------------------------------------------------------------------------

class TestRunRealWithMockEngine:
    def test_returns_one_traceresult_per_spec(self) -> None:
        cfg = {
            "name": "smoke",
            "model": {"name": "mock-model"},
            "traces": [
                {"id": "5turn", "turns": 5},
                {"id": "10turn", "turns": 10},
            ],
        }
        out = run_real(cfg, engine=MockEngine())
        assert len(out) == 2
        assert {t.trace_id for t in out} == {"5turn", "10turn"}

    def test_summary_aggregates_finite_values(self) -> None:
        cfg = {
            "name": "smoke",
            "model": {"name": "mock-model"},
            "traces": [{"id": "10turn", "turns": 10}],
        }
        out = run_real(cfg, engine=MockEngine(ttft_ms=100.0, per_token_ms=2.0))
        summary = RunSummary.from_traces(out)
        assert summary.mean_ttft_ms > 0
        assert summary.mean_ttft_ms != -1.0  # not a sentinel
        assert summary.config_name == "smoke"

    def test_accepts_explicit_specs(self) -> None:
        cfg = {"name": "smoke", "model": {"name": "mock"}}
        spec = fixture_trace(num_turns=3, tool_every=999)
        out = run_real(cfg, engine=MockEngine(), specs=[spec])
        assert len(out) == 1
        assert out[0].trace_id == spec.trace_id


# ---------------------------------------------------------------------------
# End-to-end: run_real → JSON → comparison_table.emit_table
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_full_pipeline_with_mock_engine(self, tmp_path) -> None:
        from evaluation.comparison_table import emit_table

        cfg = {
            "name": "smoke",
            "model": {"name": "mock-model"},
            "traces": [{"id": "t1", "turns": 5}, {"id": "t2", "turns": 8}],
        }
        traces = run_real(cfg, engine=MockEngine())

        # Write per-trace JSONs and one summary, the way main() does.
        for tr in traces:
            tr.emit_json(tmp_path / f"{tr.trace_id}.json")
        summary = RunSummary.from_traces(traces)

        # Round-trip: emit_table on a list with one summary works without error.
        table = emit_table([summary.__dict__])
        assert "smoke" in table
        # Real numbers, not sentinels.
        assert "-1.0" not in table.split("smoke")[1].split("\n")[0]
