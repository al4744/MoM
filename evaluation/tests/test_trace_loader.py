"""Tests for evaluation/trace_loader.py."""
from __future__ import annotations

import json

import pytest

from evaluation.trace_loader import (
    TraceSpec,
    TraceTurn,
    fixture_trace,
    load_trace,
    load_trace_directory,
)


class TestFixtureTrace:
    def test_default_shape(self) -> None:
        spec = fixture_trace()
        assert spec.trace_id == "10turn-fixture"
        assert spec.num_turns == 10

    def test_includes_tool_calls(self) -> None:
        spec = fixture_trace(num_turns=12, tool_every=3)
        assert spec.num_tool_calls > 0

    def test_post_tool_turns_present(self) -> None:
        spec = fixture_trace(num_turns=15, tool_every=3)
        assert spec.num_post_tool_turns > 0


class TestLoadTrace:
    def test_round_trip(self, tmp_path) -> None:
        spec = fixture_trace(num_turns=6, tool_every=3)
        path = tmp_path / "trace.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "trace_id": spec.trace_id,
                    "model": spec.model,
                    "prompt_tokens": spec.prompt_tokens,
                    "tool_latency_dist": spec.tool_latency_dist,
                    "turns": [t.__dict__ for t in spec.turns],
                },
                f,
            )

        loaded = load_trace(path)
        assert loaded.trace_id == spec.trace_id
        assert loaded.num_turns == spec.num_turns
        assert isinstance(loaded.turns[0], TraceTurn)


class TestLoadTraceDirectory:
    def test_loads_multiple(self, tmp_path) -> None:
        for i in range(3):
            spec = fixture_trace(trace_id=f"t{i}", num_turns=5)
            with open(tmp_path / f"t{i}.json", "w") as f:
                json.dump(
                    {
                        "trace_id": spec.trace_id,
                        "model": spec.model,
                        "prompt_tokens": spec.prompt_tokens,
                        "tool_latency_dist": spec.tool_latency_dist,
                        "turns": [t.__dict__ for t in spec.turns],
                    },
                    f,
                )
        loaded = load_trace_directory(tmp_path)
        assert len(loaded) == 3
        assert {s.trace_id for s in loaded} == {"t0", "t1", "t2"}

    def test_rejects_non_directory(self, tmp_path) -> None:
        path = tmp_path / "not_a_dir.json"
        path.write_text("{}")
        with pytest.raises(NotADirectoryError):
            load_trace_directory(path)


class TestTraceTurn:
    def test_is_tool_call(self) -> None:
        t = TraceTurn(turn_index=0, kind="tool_call", tokens=32)
        assert t.is_tool_call is True
        assert t.is_tool_return is False

    def test_is_tool_return(self) -> None:
        t = TraceTurn(turn_index=0, kind="tool_return", tokens=32)
        assert t.is_tool_return is True
