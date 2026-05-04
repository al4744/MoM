"""Workstream A — Tool-aware KV retention for vLLM v0.6.4.post1.

Pin-or-evict design: hold KV blocks for a TTL window after a tool-call turn
so the next turn can reuse them via the existing prefix-cache path.
No CPU offloading, no custom splicing.
"""
from src.retention.config import RetentionConfig, TTLConfig, PinManagerConfig, load_retention_config
from src.retention.ttl_predictor import TTLPredictor
from src.retention.pin_manager import PinManager, PinnedEntry
from src.retention import events

__all__ = [
    "RetentionConfig",
    "TTLConfig",
    "PinManagerConfig",
    "load_retention_config",
    "TTLPredictor",
    "PinManager",
    "PinnedEntry",
    "events",
]
