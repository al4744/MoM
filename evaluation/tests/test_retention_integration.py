"""Integration tests: Workstream D ↔ Workstream A retention interface.

Verifies the three integration points between D's evaluation harness and A's
retention system without requiring a GPU or a real vLLM install:

  1. program_id flows from run_trace() → engine.generate() → RequestOutput
  2. is_tool_call_pending=True is set on user_prompt turns that precede a
     tool_call turn in the trace spec
  3. build_retention_config() correctly constructs a RetentionConfig from a
     YAML-parsed dict using Workstream A's exact field names and structure
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import pytest

from evaluation.engine_adapter import MockEngine, _MockRequestOutput
from evaluation.run_eval import run_trace
from evaluation.trace_loader import TraceSpec, TraceTurn, fixture_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CapturingEngine:
    """MockEngine variant that records every generate() call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._delegate = MockEngine()

    def generate(
        self,
        prompts: Any,
        sampling_params: Any = None,
        *,
        use_tqdm: bool = False,
        program_id: Optional[str] = None,
        is_tool_call_pending: bool = False,
        tool_name: Optional[str] = None,
    ) -> list[_MockRequestOutput]:
        self.calls.append(
            dict(
                program_id=program_id,
                is_tool_call_pending=is_tool_call_pending,
                tool_name=tool_name,
            )
        )
        return self._delegate.generate(
            prompts,
            sampling_params,
            use_tqdm=use_tqdm,
            program_id=program_id,
            is_tool_call_pending=is_tool_call_pending,
            tool_name=tool_name,
        )


# ---------------------------------------------------------------------------
# 1. program_id flows end-to-end
# ---------------------------------------------------------------------------

class TestProgramIdFlow:
    def test_program_id_is_set_on_every_call(self) -> None:
        engine = _CapturingEngine()
        spec = fixture_trace(num_turns=4, tool_every=999)  # no tools
        run_trace(engine, spec, cfg={"name": "test"})
        assert len(engine.calls) > 0
        assert all(c["program_id"] is not None for c in engine.calls)

    def test_program_id_is_consistent_across_turns(self) -> None:
        engine = _CapturingEngine()
        spec = fixture_trace(num_turns=6, tool_every=999)
        run_trace(engine, spec, cfg={"name": "test"})
        pids = [c["program_id"] for c in engine.calls]
        assert len(set(pids)) == 1, "all turns in a trace must share one program_id"

    def test_different_traces_get_different_program_ids(self) -> None:
        engine_a = _CapturingEngine()
        engine_b = _CapturingEngine()
        spec = fixture_trace(num_turns=3, tool_every=999)
        run_trace(engine_a, spec, cfg={"name": "a"})
        run_trace(engine_b, spec, cfg={"name": "b"})
        pid_a = engine_a.calls[0]["program_id"]
        pid_b = engine_b.calls[0]["program_id"]
        assert pid_a != pid_b

    def test_program_id_echoed_on_mock_output(self) -> None:
        engine = MockEngine()
        outputs = engine.generate("hello", program_id="test-pid-123")
        assert outputs[0].program_id == "test-pid-123"

    def test_mock_request_output_has_program_id_field(self) -> None:
        out = _MockRequestOutput(
            request_id="r1",
            prompt="p",
            metrics=SimpleNamespace(  # type: ignore[arg-type]
                arrival_time=0.0,
                first_scheduled_time=0.0,
                first_token_time=0.1,
                last_token_time=0.2,
            ),
            outputs=[],
            program_id="prog-42",
        )
        assert out.program_id == "prog-42"

    def test_mock_request_output_program_id_defaults_to_none(self) -> None:
        engine = MockEngine()
        outputs = engine.generate("hi", None)
        assert outputs[0].program_id is None


# ---------------------------------------------------------------------------
# 2. is_tool_call_pending set correctly
# ---------------------------------------------------------------------------

class TestIsToolCallPending:
    def _run_and_capture(self, spec: TraceSpec) -> list[dict[str, Any]]:
        engine = _CapturingEngine()
        run_trace(engine, spec, cfg={"name": "test"})
        return engine.calls

    def test_no_tools_means_never_pending(self) -> None:
        spec = fixture_trace(num_turns=5, tool_every=999)
        calls = self._run_and_capture(spec)
        assert all(not c["is_tool_call_pending"] for c in calls)

    def test_user_prompt_before_tool_call_is_pending(self) -> None:
        # Explicit spec: user_prompt → tool_call → tool_return → user_prompt
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
        calls = self._run_and_capture(spec)
        # Two user_prompt turns → two generate() calls
        assert len(calls) == 2
        # First user_prompt (index 0) precedes a tool_call → pending
        assert calls[0]["is_tool_call_pending"] is True
        # Second user_prompt (index 3) has nothing after it → not pending
        assert calls[1]["is_tool_call_pending"] is False

    def test_tool_name_forwarded_when_pending(self) -> None:
        spec = TraceSpec(
            trace_id="tn",
            model="mock",
            prompt_tokens=64,
            tool_latency_dist="zero",
            turns=[
                TraceTurn(turn_index=0, kind="user_prompt", tokens=32),
                TraceTurn(turn_index=1, kind="tool_call", tokens=8,
                          tool_name="pytest", tool_latency_ms=0.0),
                TraceTurn(turn_index=2, kind="tool_return", tokens=16,
                          tool_name="pytest"),
            ],
        )
        calls = self._run_and_capture(spec)
        assert calls[0]["tool_name"] == "pytest"

    def test_tool_name_none_when_not_pending(self) -> None:
        spec = fixture_trace(num_turns=3, tool_every=999)
        calls = self._run_and_capture(spec)
        assert all(c["tool_name"] is None for c in calls)

    def test_multiple_tool_calls_all_marked(self) -> None:
        # u tc tr u tc tr u  — two tool-call-pending turns
        spec = TraceSpec(
            trace_id="multi",
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
                TraceTurn(turn_index=4, kind="tool_call", tokens=8,
                          tool_name="pytest", tool_latency_ms=0.0),
                TraceTurn(turn_index=5, kind="tool_return", tokens=16,
                          tool_name="pytest"),
                TraceTurn(turn_index=6, kind="user_prompt", tokens=32),
            ],
        )
        calls = self._run_and_capture(spec)
        assert len(calls) == 3
        assert calls[0]["is_tool_call_pending"] is True   # before search
        assert calls[1]["is_tool_call_pending"] is True   # before pytest
        assert calls[2]["is_tool_call_pending"] is False  # last turn


# ---------------------------------------------------------------------------
# 3. build_retention_config constructs correct Workstream A types
# ---------------------------------------------------------------------------

class TestBuildRetentionConfig:
    def test_returns_none_when_disabled(self) -> None:
        from evaluation.engine_adapter import build_retention_config
        assert build_retention_config({}) is None
        assert build_retention_config({"enabled": False}) is None

    def test_returns_retention_config_when_enabled(self) -> None:
        from evaluation.engine_adapter import build_retention_config
        from src.retention.config import RetentionConfig
        cfg = build_retention_config({"enabled": True})
        assert isinstance(cfg, RetentionConfig)
        assert cfg.enabled is True

    def test_ttl_fields_map_correctly(self) -> None:
        from evaluation.engine_adapter import build_retention_config
        cfg = build_retention_config({
            "enabled": True,
            "ttl": {
                "alpha": 0.2,
                "default_ttl": 2.0,
                "safety_factor": 1.8,
                "use_per_tool_ema": False,
                "use_ema": True,
            },
        })
        assert cfg.ttl.alpha == pytest.approx(0.2)
        assert cfg.ttl.default_ttl == pytest.approx(2.0)
        assert cfg.ttl.safety_factor == pytest.approx(1.8)
        assert cfg.ttl.use_per_tool_ema is False
        assert cfg.ttl.use_ema is True

    def test_pin_manager_fields_map_correctly(self) -> None:
        from evaluation.engine_adapter import build_retention_config
        cfg = build_retention_config({
            "enabled": True,
            "pin_manager": {"max_pinned_fraction": 0.25},
        })
        assert cfg.pin_manager.max_pinned_fraction == pytest.approx(0.25)

    def test_unknown_ttl_fields_are_silently_dropped(self) -> None:
        from evaluation.engine_adapter import build_retention_config
        # Fields from the old schema (int4/int8 configs) must not raise.
        cfg = build_retention_config({
            "enabled": True,
            "ttl": {
                "ema_alpha": 0.2,       # wrong name — should be dropped
                "default_ms": 1500,     # wrong name + wrong unit — should be dropped
                "min_samples": 3,       # doesn't exist in TTLConfig
                "cap_ms": 10000,        # doesn't exist in TTLConfig
            },
        })
        # All invalid keys dropped → falls back to TTLConfig defaults
        from src.retention.config import TTLConfig
        defaults = TTLConfig()
        assert cfg.ttl.alpha == defaults.alpha
        assert cfg.ttl.default_ttl == defaults.default_ttl

    def test_retention_yaml_config_parses_correctly(self) -> None:
        """End-to-end: load configs/retention.yaml and build a RetentionConfig."""
        from pathlib import Path
        import yaml
        from evaluation.engine_adapter import build_retention_config

        yaml_path = Path(__file__).parent.parent.parent / "configs" / "retention.yaml"
        if not yaml_path.exists():
            pytest.skip("configs/retention.yaml not found")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        retention_dict = raw.get("engine", {}).get("retention", {})
        cfg = build_retention_config(retention_dict)
        assert cfg is not None
        assert cfg.enabled is True
