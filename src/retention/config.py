from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


def _from_dict(cls: type, data: Dict[str, Any]) -> Any:
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in valid})


@dataclass
class TTLConfig:
    alpha: float = 0.3
    default_ttl: float = 1.0
    safety_factor: float = 1.5
    # Ablation: False → global EMA only (no per-tool tracking)
    use_per_tool_ema: bool = True
    # Ablation: False → constant default_ttl * safety_factor always
    use_ema: bool = True


@dataclass
class PinManagerConfig:
    max_pinned_fraction: float = 0.3


@dataclass
class RetentionConfig:
    enabled: bool = True
    ttl: TTLConfig = field(default_factory=TTLConfig)
    pin_manager: PinManagerConfig = field(default_factory=PinManagerConfig)


def load_retention_config(path: str) -> RetentionConfig:
    """Load retention config from a YAML file.

    Expected schema:
        retention:
          enabled: true
          ttl:
            alpha: 0.3
            default_ttl: 1.0
            safety_factor: 1.5
            use_per_tool_ema: true
            use_ema: true
          pin_manager:
            max_pinned_fraction: 0.3
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    r = raw.get("retention", {})
    return RetentionConfig(
        enabled=r.get("enabled", True),
        ttl=_from_dict(TTLConfig, r.get("ttl", {})),
        pin_manager=_from_dict(PinManagerConfig, r.get("pin_manager", {})),
    )


def default_retention_config() -> RetentionConfig:
    return RetentionConfig()
