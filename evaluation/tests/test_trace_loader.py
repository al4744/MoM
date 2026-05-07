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


class TestRoleField:
    """role= field on TraceSpec — controls focal/filler scheduling in
    concurrent_runner."""

    def test_default_role_is_focal(self) -> None:
        spec = fixture_trace()
        assert spec.role == "focal"

    def test_role_filler_propagates(self) -> None:
        spec = fixture_trace(role="filler")
        assert spec.role == "filler"

    def test_tracespec_default_role_focal(self) -> None:
        # Direct construction without role kwarg (e.g. legacy callers).
        spec = TraceSpec(
            trace_id="t",
            model="m",
            prompt_tokens=128,
            tool_latency_dist="zero",
        )
        assert spec.role == "focal"


class TestFillerFocalWorkload:
    """trace_loader.filler_focal_workload — Daksh's microbenchmark builder."""

    def test_one_focal_n_fillers(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(num_fillers=3, focal_num_user_prompts=2)
        assert len(specs) == 4
        assert specs[0].role == "focal"
        assert all(s.role == "filler" for s in specs[1:])

    def test_focal_user_prompts_match_argument(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        for n in (1, 4, 10):
            specs = filler_focal_workload(num_fillers=0, focal_num_user_prompts=n)
            user_prompts = [t for t in specs[0].turns if t.kind == "user_prompt"]
            assert len(user_prompts) == n

    def test_filler_turns_are_all_user_prompts(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(num_fillers=2, filler_num_turns=10)
        for filler in specs[1:]:
            assert all(t.kind == "user_prompt" for t in filler.turns)
            assert len(filler.turns) == 10

    def test_focal_tool_latency_in_each_gap(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=0,
            focal_num_user_prompts=3,
            focal_tool_latency_ms=4321.0,
        )
        tool_calls = [t for t in specs[0].turns if t.kind == "tool_call"]
        assert all(tc.tool_latency_ms == 4321.0 for tc in tool_calls)

    def test_body_tokens_match(self) -> None:
        from evaluation.trace_loader import filler_focal_workload
        specs = filler_focal_workload(
            num_fillers=2,
            focal_num_user_prompts=2,
            focal_body_tokens_per_turn=999,
            filler_body_tokens_per_turn=111,
        )
        focal_user_prompts = [t for t in specs[0].turns if t.kind == "user_prompt"]
        filler_user_prompts = [t for t in specs[1].turns if t.kind == "user_prompt"]
        assert all(t.tokens == 999 for t in focal_user_prompts)
        assert all(t.tokens == 111 for t in filler_user_prompts)
