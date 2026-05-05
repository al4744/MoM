"""Tests for evaluation/engine_adapter.py — MockEngine + helpers."""
from __future__ import annotations

import pytest

from evaluation.engine_adapter import (
    MockEngine,
    extract_request_metrics,
    output_token_count,
)


class TestMockEngineGenerate:
    def test_returns_one_output_per_prompt(self) -> None:
        engine = MockEngine()
        outputs = engine.generate(["hello", "world"], None)
        assert len(outputs) == 2

    def test_handles_single_string_prompt(self) -> None:
        engine = MockEngine()
        outputs = engine.generate("hello", None)
        assert len(outputs) == 1

    def test_request_id_is_unique(self) -> None:
        engine = MockEngine()
        a = engine.generate("hi", None)[0].request_id
        b = engine.generate("hi", None)[0].request_id
        assert a != b

    def test_metrics_have_arrival_and_first_token(self) -> None:
        engine = MockEngine()
        out = engine.generate("hi", None)[0]
        assert out.metrics.arrival_time > 0
        assert out.metrics.first_token_time is not None
        assert out.metrics.first_token_time > out.metrics.arrival_time

    def test_first_token_offset_matches_ttft_ms(self) -> None:
        engine = MockEngine(ttft_ms=200.0, per_token_ms=0.0)
        out = engine.generate("hi", None)[0]
        delta_s = out.metrics.first_token_time - out.metrics.arrival_time
        # Synthetic timing should be exact (no real sleep).
        assert delta_s == pytest.approx(0.2, abs=1e-6)

    def test_decode_window_matches_per_token_ms(self) -> None:
        engine = MockEngine(ttft_ms=0.0, per_token_ms=10.0, max_output_tokens=8)
        # Provide a sampling_params dict with max_tokens=8 so we get 8 tokens.
        out = engine.generate("hi", {"max_tokens": 8})[0]
        delta_s = out.metrics.last_token_time - out.metrics.first_token_time
        assert delta_s == pytest.approx(0.08, abs=1e-6)  # 8 * 10ms

    def test_respects_sampling_params_max_tokens(self) -> None:
        engine = MockEngine(max_output_tokens=100)
        # max_tokens=4 in sampling params should clamp output to 4 tokens.
        out = engine.generate("hi", {"max_tokens": 4})[0]
        assert len(out.outputs[0].token_ids) == 4

    def test_real_time_simulation_actually_sleeps(self) -> None:
        import time as _time

        engine = MockEngine(
            ttft_ms=20.0,
            per_token_ms=2.0,
            max_output_tokens=4,
            simulate_realtime=True,
        )
        t0 = _time.monotonic()
        engine.generate("hi", {"max_tokens": 4})
        elapsed_ms = (_time.monotonic() - t0) * 1000.0
        # Floor: 20 + 4*2 == 28ms; allow generous slack.
        assert elapsed_ms >= 25.0


class TestExtractRequestMetrics:
    def test_computes_ttft_ms(self) -> None:
        engine = MockEngine(ttft_ms=150.0, per_token_ms=0.0)
        out = engine.generate("hi", None)[0]
        m = extract_request_metrics(out)
        assert m["ttft_ms"] == pytest.approx(150.0, abs=1e-3)

    def test_handles_missing_first_token(self) -> None:
        # Build a fake output with first_token_time=None — exercises the None branch.
        from evaluation.engine_adapter import (
            _MockCompletionOutput,
            _MockMetrics,
            _MockRequestOutput,
        )
        out = _MockRequestOutput(
            request_id="x",
            prompt="x",
            metrics=_MockMetrics(
                arrival_time=0.0,
                first_scheduled_time=None,
                first_token_time=None,
                last_token_time=0.0,
            ),
            outputs=[_MockCompletionOutput(text="", token_ids=[])],
        )
        m = extract_request_metrics(out)
        assert m["ttft_ms"] is None
        assert m["decode_ms"] is None


class TestOutputTokenCount:
    def test_counts_tokens(self) -> None:
        engine = MockEngine(max_output_tokens=7)
        out = engine.generate("hi", {"max_tokens": 7})[0]
        assert output_token_count(out) == 7

    def test_zero_when_outputs_missing(self) -> None:
        class _Empty:
            outputs = []
        assert output_token_count(_Empty()) == 0
