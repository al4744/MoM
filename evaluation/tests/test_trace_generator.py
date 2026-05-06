from __future__ import annotations

import argparse

from benchmarks.trace_generator import build_trace, write_traces
from evaluation.trace_loader import load_trace


def test_build_trace_matches_loader_schema() -> None:
    trace = build_trace(
        trace_id="1k-5turn-mixed",
        model="mock",
        turns=5,
        context_length=1024,
        tool_latency_dist="mixed",
        tool_every=3,
        expected_output_tokens=16,
    )
    assert trace["trace_id"] == "1k-5turn-mixed"
    assert trace["prompt_tokens"] == 1024
    assert len(trace["turns"]) == 5
    assert any(turn["kind"] == "tool_call" for turn in trace["turns"])


def test_write_traces_round_trips(tmp_path) -> None:
    args = argparse.Namespace(
        output=tmp_path,
        turns=[5],
        context_lengths=[1024],
        tool_latency_dist="mixed",
        model="mock",
        tool_every=3,
        expected_output_tokens=8,
    )
    paths = write_traces(args)
    assert len(paths) == 1
    loaded = load_trace(paths[0])
    assert loaded.trace_id == "1k-5turn-mixed"
    assert loaded.model == "mock"
    assert loaded.prompt_tokens == 1024
