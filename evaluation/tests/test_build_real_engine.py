"""Tests for evaluation/engine_adapter.build_real_engine kwargs forwarding.

These tests mock out ``vllm.LLM`` so they run on CPU without a GPU or a real
vLLM install. They verify the exact kwargs that build_real_engine() passes to
the LLM constructor — the category of bug that passes all MockEngine-based
tests but crashes on a real GPU run because a required kwarg was missing or
named wrong.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_build(cfg: dict) -> dict:
    """Call build_real_engine with a patched LLM; return the kwargs it received."""
    captured: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("evaluation.engine_adapter.LLM", _FakeLLM, create=True), \
         patch("builtins.__import__", _selective_import(_FakeLLM)):
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(cfg)

    return captured


def _selective_import(fake_llm_cls):
    """Return an __import__ that replaces ``vllm.LLM`` with fake_llm_cls."""
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") \
        else __import__

    def _import(name, *args, **kwargs):
        if name == "vllm":
            mod = MagicMock()
            mod.LLM = fake_llm_cls
            return mod
        return real_import(name, *args, **kwargs)

    return _import


def _baseline_cfg(**engine_overrides) -> dict:
    base = {
        "model": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
        },
        "engine": {
            "retention": {"enabled": False},
            "prefix_caching": {"enabled": False},
        },
    }
    base["engine"].update(engine_overrides)
    return base


def _retention_cfg(**ttl_overrides) -> dict:
    ttl = {"alpha": 0.3, "default_ttl": 1.0, "safety_factor": 1.5,
           "use_per_tool_ema": True, "use_ema": True}
    ttl.update(ttl_overrides)
    return {
        "model": {
            "name": "meta-llama/Meta-Llama-3-8B",
            "dtype": "float16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
        },
        "engine": {
            "retention": {
                "enabled": True,
                "ttl": ttl,
                "pin_manager": {"max_pinned_fraction": 0.3},
            },
            "prefix_caching": {"enabled": True},
        },
    }


# ---------------------------------------------------------------------------
# Fixture to patch LLM at the right import location
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_llm():
    """Patch vllm.LLM so build_real_engine() can be tested without vLLM."""
    captured_kwargs: dict = {}

    class _FakeLLM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    with patch.dict("sys.modules", {"vllm": MagicMock(LLM=_FakeLLM)}):
        yield captured_kwargs


# ---------------------------------------------------------------------------
# Model kwargs always forwarded
# ---------------------------------------------------------------------------

class TestModelKwargs:
    def test_model_name_forwarded(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert fake_llm["model"] == "meta-llama/Meta-Llama-3-8B"

    def test_dtype_forwarded(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert fake_llm["dtype"] == "float16"

    def test_max_model_len_forwarded(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert fake_llm["max_model_len"] == 4096

    def test_gpu_memory_utilization_forwarded(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert fake_llm["gpu_memory_utilization"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# prefix_caching forwarded correctly
# ---------------------------------------------------------------------------

class TestPrefixCaching:
    def test_baseline_has_prefix_caching_false(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert fake_llm["enable_prefix_caching"] is False

    def test_retention_has_prefix_caching_true(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_retention_cfg())
        assert fake_llm["enable_prefix_caching"] is True

    def test_missing_prefix_caching_key_defaults_to_false(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        cfg = _baseline_cfg()
        del cfg["engine"]["prefix_caching"]
        build_real_engine(cfg)
        assert fake_llm["enable_prefix_caching"] is False


# ---------------------------------------------------------------------------
# retention_config forwarded correctly
# ---------------------------------------------------------------------------

class TestRetentionConfig:
    def test_baseline_has_no_retention_config(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_baseline_cfg())
        assert "retention_config" not in fake_llm

    def test_retention_config_present_when_enabled(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        from src.retention.config import RetentionConfig
        build_real_engine(_retention_cfg())
        assert "retention_config" in fake_llm
        assert isinstance(fake_llm["retention_config"], RetentionConfig)

    def test_retention_config_ttl_alpha(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_retention_cfg())
        assert fake_llm["retention_config"].ttl.alpha == pytest.approx(0.3)

    def test_retention_config_pin_fraction(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(_retention_cfg())
        assert fake_llm["retention_config"].pin_manager.max_pinned_fraction \
            == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Real YAML configs produce correct kwargs
# ---------------------------------------------------------------------------

class TestYamlConfigsProduceCorrectKwargs:
    def _load(self, name: str) -> dict:
        from pathlib import Path
        import yaml
        p = Path(__file__).parent.parent.parent / "configs" / f"{name}.yaml"
        if not p.exists():
            pytest.skip(f"configs/{name}.yaml not found")
        with open(p) as f:
            return yaml.safe_load(f)

    def test_baseline_yaml_prefix_caching_false(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(self._load("baseline"))
        assert fake_llm["enable_prefix_caching"] is False

    def test_retention_yaml_prefix_caching_true(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(self._load("retention"))
        assert fake_llm["enable_prefix_caching"] is True

    def test_retention_yaml_has_retention_config(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        from src.retention.config import RetentionConfig
        build_real_engine(self._load("retention"))
        assert isinstance(fake_llm.get("retention_config"), RetentionConfig)

    def test_baseline_yaml_no_retention_config(self, fake_llm) -> None:
        from evaluation.engine_adapter import build_real_engine
        build_real_engine(self._load("baseline"))
        assert "retention_config" not in fake_llm


# ---------------------------------------------------------------------------
# Preemption counter reads correct attribute from correct object
# ---------------------------------------------------------------------------

class TestPreemptionCounter:
    def test_reads_num_cumulative_preemption_not_num_preemption(self) -> None:
        """Regression: old code checked num_preemption which doesn't exist."""
        from evaluation.engine_adapter import MockEngine
        from evaluation.run_eval import run_trace
        from evaluation.trace_loader import fixture_trace

        engine = MockEngine()

        # Attach a fake llm_engine with the real attribute name on a list.
        fake_scheduler = MagicMock()
        fake_scheduler.num_cumulative_preemption = 7
        fake_scheduler.num_preemption = MagicMock()  # wrong attr — should NOT be read

        fake_llm_engine = MagicMock()
        fake_llm_engine.scheduler = [fake_scheduler]
        engine.llm_engine = fake_llm_engine

        spec = fixture_trace(num_turns=3, tool_every=999)
        result = run_trace(engine, spec, cfg={"name": "t"})
        assert result.preemption_count == 7

    def test_sums_across_pipeline_stages(self) -> None:
        from evaluation.engine_adapter import MockEngine
        from evaluation.run_eval import run_trace
        from evaluation.trace_loader import fixture_trace

        engine = MockEngine()
        s1, s2 = MagicMock(), MagicMock()
        s1.num_cumulative_preemption = 3
        s2.num_cumulative_preemption = 5

        engine.llm_engine = MagicMock()
        engine.llm_engine.scheduler = [s1, s2]

        spec = fixture_trace(num_turns=2, tool_every=999)
        result = run_trace(engine, spec, cfg={"name": "t"})
        assert result.preemption_count == 8

    def test_zero_when_no_llm_engine(self) -> None:
        from evaluation.engine_adapter import MockEngine
        from evaluation.run_eval import run_trace
        from evaluation.trace_loader import fixture_trace

        engine = MockEngine()  # no llm_engine attribute
        spec = fixture_trace(num_turns=2, tool_every=999)
        result = run_trace(engine, spec, cfg={"name": "t"})
        assert result.preemption_count == 0
