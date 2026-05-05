"""Tests for evaluation/engine_adapter.py — MockEngine + helpers."""
from __future__ import annotations

import pytest

from evaluation.engine_adapter import (
    MockEngine,
    build_retention_config,
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


class TestBuildRetentionConfig:
    """Verify the YAML → src.retention.config.RetentionConfig translation."""

    def test_returns_none_for_disabled(self) -> None:
        assert build_retention_config({"enabled": False}) is None

    def test_returns_none_for_empty_dict(self) -> None:
        assert build_retention_config({}) is None

    def test_returns_none_for_missing_enabled(self) -> None:
        assert build_retention_config({"ttl": {"alpha": 0.3}}) is None

    def test_constructs_with_full_yaml(self) -> None:
        from src.retention.config import (
            PinManagerConfig,
            RetentionConfig,
            TTLConfig,
        )
        cfg = build_retention_config(
            {
                "enabled": True,
                "ttl": {
                    "alpha": 0.5,
                    "default_ttl": 2.0,
                    "safety_factor": 1.8,
                    "use_per_tool_ema": False,
                    "use_ema": True,
                },
                "pin_manager": {"max_pinned_fraction": 0.4},
            }
        )
        assert isinstance(cfg, RetentionConfig)
        assert cfg.enabled is True
        assert isinstance(cfg.ttl, TTLConfig)
        assert cfg.ttl.alpha == 0.5
        assert cfg.ttl.default_ttl == 2.0
        assert cfg.ttl.safety_factor == 1.8
        assert cfg.ttl.use_per_tool_ema is False
        assert cfg.ttl.use_ema is True
        assert isinstance(cfg.pin_manager, PinManagerConfig)
        assert cfg.pin_manager.max_pinned_fraction == 0.4

    def test_uses_dataclass_defaults_for_missing_fields(self) -> None:
        from src.retention.config import RetentionConfig
        # Only specify enabled — every other field should fall back to dataclass defaults.
        cfg = build_retention_config({"enabled": True})
        assert isinstance(cfg, RetentionConfig)
        assert cfg.ttl.alpha == 0.3            # TTLConfig default
        assert cfg.pin_manager.max_pinned_fraction == 0.3  # PinManagerConfig default

    def test_ignores_unknown_yaml_keys(self) -> None:
        # If we ever add fields to retention.yaml that don't exist in the
        # dataclass, build_retention_config should drop them rather than crash.
        from src.retention.config import RetentionConfig
        cfg = build_retention_config(
            {
                "enabled": True,
                "ttl": {"alpha": 0.4, "future_field_we_havent_added": 42},
                "pin_manager": {"max_pinned_fraction": 0.25, "noise": "ok"},
            }
        )
        assert isinstance(cfg, RetentionConfig)
        assert cfg.ttl.alpha == 0.4
        assert cfg.pin_manager.max_pinned_fraction == 0.25


class TestRetentionYamlEndToEnd:
    """The actual configs/retention.yaml file should be loadable into a
    RetentionConfig without modification. Catches schema drift between the
    YAML committed to the repo and Daksh's dataclass."""

    def test_committed_yaml_parses(self) -> None:
        import yaml
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        with open(repo_root / "configs" / "retention.yaml") as f:
            cfg = yaml.safe_load(f)

        retention = build_retention_config(
            cfg.get("engine", {}).get("retention", {})
        )
        assert retention is not None
        assert retention.enabled is True
        assert 0 < retention.pin_manager.max_pinned_fraction <= 1.0
