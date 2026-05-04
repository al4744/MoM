from __future__ import annotations

from typing import Optional


class TTLPredictor:
    """Per-tool EMA TTL predictor with global-EMA and fixed-default fallbacks.

    Fallback hierarchy on predict_ttl():
        1. per-tool EMA  (if use_per_tool_ema=True and tool has been observed)
        2. global EMA    (if use_ema=True and any tool has been observed)
        3. default_ttl   (cold start)

    Ablation flags
    --------------
    use_ema : bool
        False → always return default_ttl * safety_factor, no EMA involved.
    use_per_tool_ema : bool
        False → only global EMA is maintained; per-tool entries are skipped.
        Ignored when use_ema=False.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        default_ttl: float = 1.0,
        safety_factor: float = 1.5,
        use_per_tool_ema: bool = True,
        use_ema: bool = True,
    ) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {default_ttl}")
        if safety_factor <= 0:
            raise ValueError(f"safety_factor must be positive, got {safety_factor}")

        self.alpha = alpha
        self.default_ttl = default_ttl
        self.safety_factor = safety_factor
        self.use_per_tool_ema = use_per_tool_ema
        self.use_ema = use_ema

        self.per_tool_ema: dict[str, float] = {}
        self.global_ema: Optional[float] = None

    def predict_ttl(self, tool_name: str) -> float:
        """Return predicted TTL (seconds) for the given tool call."""
        if not self.use_ema:
            return self.default_ttl * self.safety_factor

        if self.use_per_tool_ema and tool_name in self.per_tool_ema:
            estimate = self.per_tool_ema[tool_name]
        elif self.global_ema is not None:
            estimate = self.global_ema
        else:
            estimate = self.default_ttl

        return estimate * self.safety_factor

    def update(self, tool_name: str, observed_duration: float) -> None:
        """Update EMAs with a newly observed tool-call round-trip duration."""
        if not self.use_ema:
            return

        if self.use_per_tool_ema:
            if tool_name in self.per_tool_ema:
                self.per_tool_ema[tool_name] = (
                    self.alpha * observed_duration
                    + (1 - self.alpha) * self.per_tool_ema[tool_name]
                )
            else:
                self.per_tool_ema[tool_name] = observed_duration

        if self.global_ema is None:
            self.global_ema = observed_duration
        else:
            self.global_ema = (
                self.alpha * observed_duration
                + (1 - self.alpha) * self.global_ema
            )

    @classmethod
    def from_config(cls, cfg: "TTLConfig") -> "TTLPredictor":  # noqa: F821
        """Construct from a TTLConfig dataclass."""
        return cls(
            alpha=cfg.alpha,
            default_ttl=cfg.default_ttl,
            safety_factor=cfg.safety_factor,
            use_per_tool_ema=cfg.use_per_tool_ema,
            use_ema=cfg.use_ema,
        )
