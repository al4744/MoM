"""Workstream D — End-to-end evaluation.

Metrics suite, eval runner, and comparison table generator. See:
  - evaluation/metrics.py        — TimingContext, TurnMetrics, TraceResult, RunSummary
  - evaluation/run_eval.py       — CLI: run one config across all its traces
  - evaluation/comparison_table  — CLI: emit markdown comparison across configs
  - evaluation/README.md         — methodology + metric definitions
"""
from evaluation.metrics import (
    RunSummary,
    TimingContext,
    TraceResult,
    TurnMetrics,
)

__all__ = ["RunSummary", "TimingContext", "TraceResult", "TurnMetrics"]
