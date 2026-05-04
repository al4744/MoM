import pytest

from src.retention.ttl_predictor import TTLPredictor


class TestColdStart:
    def test_returns_default_scaled_by_safety(self):
        p = TTLPredictor(alpha=0.3, default_ttl=2.0, safety_factor=1.5)
        assert p.predict_ttl("pytest") == pytest.approx(3.0)

    def test_unknown_tool_uses_default(self):
        p = TTLPredictor(default_ttl=1.0, safety_factor=2.0)
        assert p.predict_ttl("unknown_tool") == pytest.approx(2.0)


class TestFirstUpdate:
    def test_per_tool_ema_initialized_to_observed(self):
        p = TTLPredictor(alpha=0.3, default_ttl=1.0, safety_factor=1.0)
        p.update("pytest", 5.0)
        assert p.per_tool_ema["pytest"] == pytest.approx(5.0)

    def test_global_ema_initialized_to_observed(self):
        p = TTLPredictor(alpha=0.3)
        p.update("web_search", 3.0)
        assert p.global_ema == pytest.approx(3.0)

    def test_predict_uses_per_tool_after_first_update(self):
        p = TTLPredictor(alpha=0.3, safety_factor=1.0)
        p.update("pytest", 4.0)
        assert p.predict_ttl("pytest") == pytest.approx(4.0)


class TestEMABlending:
    def test_second_update_blends_correctly(self):
        p = TTLPredictor(alpha=0.5, default_ttl=1.0, safety_factor=1.0)
        p.update("pytest", 4.0)   # EMA = 4.0
        p.update("pytest", 2.0)   # EMA = 0.5*2 + 0.5*4 = 3.0
        assert p.per_tool_ema["pytest"] == pytest.approx(3.0)

    def test_global_ema_blends_across_tools(self):
        p = TTLPredictor(alpha=0.5, safety_factor=1.0)
        p.update("tool_a", 4.0)   # global = 4.0
        p.update("tool_b", 2.0)   # global = 0.5*2 + 0.5*4 = 3.0
        assert p.global_ema == pytest.approx(3.0)

    def test_unknown_tool_falls_back_to_global_after_any_update(self):
        p = TTLPredictor(alpha=0.3, safety_factor=1.0)
        p.update("tool_a", 6.0)
        assert p.predict_ttl("new_tool") == pytest.approx(6.0)


class TestAblationUseEmaFalse:
    def test_predict_always_returns_constant(self):
        p = TTLPredictor(default_ttl=2.0, safety_factor=1.5, use_ema=False)
        p.update("pytest", 99.0)  # should have no effect
        assert p.predict_ttl("pytest") == pytest.approx(3.0)

    def test_update_does_not_populate_ema(self):
        p = TTLPredictor(use_ema=False)
        p.update("pytest", 99.0)
        assert p.global_ema is None
        assert not p.per_tool_ema


class TestAblationUsePerToolEmaFalse:
    def test_update_does_not_populate_per_tool(self):
        p = TTLPredictor(use_per_tool_ema=False)
        p.update("pytest", 5.0)
        assert not p.per_tool_ema

    def test_predict_uses_global_ema_only(self):
        p = TTLPredictor(alpha=1.0, safety_factor=1.0, use_per_tool_ema=False)
        p.update("pytest", 7.0)
        # global EMA = 7.0; per-tool empty
        assert p.predict_ttl("pytest") == pytest.approx(7.0)
        assert p.predict_ttl("any_other_tool") == pytest.approx(7.0)


class TestSafetyFactor:
    def test_safety_factor_scales_prediction(self):
        p = TTLPredictor(alpha=1.0, safety_factor=2.0)
        p.update("t", 3.0)
        assert p.predict_ttl("t") == pytest.approx(6.0)


class TestValidation:
    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError):
            TTLPredictor(alpha=0.0)

    def test_alpha_above_one_raises(self):
        with pytest.raises(ValueError):
            TTLPredictor(alpha=1.1)

    def test_nonpositive_default_ttl_raises(self):
        with pytest.raises(ValueError):
            TTLPredictor(default_ttl=0.0)

    def test_nonpositive_safety_factor_raises(self):
        with pytest.raises(ValueError):
            TTLPredictor(safety_factor=-1.0)


class TestFromConfig:
    def test_from_config_roundtrip(self):
        from src.retention.config import TTLConfig
        cfg = TTLConfig(alpha=0.2, default_ttl=3.0, safety_factor=2.0,
                        use_per_tool_ema=False, use_ema=True)
        p = TTLPredictor.from_config(cfg)
        assert p.alpha == 0.2
        assert p.default_ttl == 3.0
        assert p.safety_factor == 2.0
        assert not p.use_per_tool_ema
        assert p.use_ema
