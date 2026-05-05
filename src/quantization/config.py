from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


QuantMode = Literal["int8", "int4"]


@dataclass(frozen=True)
class KVQuantConfig:
    enabled: bool = False
    mode: QuantMode = "int8"
    group_size: int = 64
    per_channel: bool = False
    recent_turns_fp: int = 2
    max_error_guard: float = 0.12


def load_kv_quant_config(cfg: Mapping[str, Any] | None) -> KVQuantConfig:
    if not cfg:
        return KVQuantConfig(enabled=False)
    mode = cfg.get("kv_cache")
    if mode not in ("int8", "int4"):
        return KVQuantConfig(enabled=False)
    return KVQuantConfig(
        enabled=True,
        mode=mode,
        group_size=int(cfg.get("group_size", 64)),
        per_channel=bool(cfg.get("per_channel", False)),
        recent_turns_fp=int(cfg.get("recent_turns_fp", 2)),
        max_error_guard=float(cfg.get("max_error_guard", 0.12)),
    )
