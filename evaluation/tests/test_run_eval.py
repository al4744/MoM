"""Tests for evaluation/run_eval.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import run_eval


class TestStubTrace:
    def test_generates_expected_turn_count(self) -> None:
        tr = run_eval.stub_trace("baseline", "t1", num_turns=7)
        assert len(tr.turns) == 7

    def test_marks_post_tool_turns(self) -> None:
        tr = run_eval.stub_trace("baseline", "t1", num_turns=10)
        post_tool = [t for t in tr.turns if t.prefill_recomp_ms is not None]
        # Stub marks i > 0 and i % 3 == 0; that's indices 3, 6, 9 → 3 turns.
        assert len(post_tool) == 3

    def test_emits_sentinel_values(self) -> None:
        tr = run_eval.stub_trace("baseline", "t1", num_turns=3)
        # Sentinel -1.0 marks dry-run output so it cannot be silently confused
        # with real measurements.
        assert tr.peak_vram_mb == -1.0
        assert tr.preemption_count == -1
        assert tr.metadata.get("stub") is True
        assert all(t.ttft_ms == -1.0 for t in tr.turns)


class TestRunReal:
    def test_raises_until_engine_hooks_land(self) -> None:
        with pytest.raises(NotImplementedError):
            run_eval.run_real({})


class TestLoadConfig:
    def test_loads_yaml_config(self, tmp_path) -> None:
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("name: test\ntraces:\n  - id: t1\n    turns: 5\n")
        cfg = run_eval.load_config(cfg_path)
        assert cfg["name"] == "test"
        assert cfg["traces"][0]["turns"] == 5


class TestDryRunEndToEnd:
    """Exercise the full main() path end-to-end in dry-run mode."""

    def test_writes_per_trace_and_summary(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "test.yaml"
        cfg.write_text(
            "name: test\n"
            "traces:\n"
            "  - id: t1\n    turns: 5\n"
            "  - id: t2\n    turns: 10\n"
        )
        outdir = tmp_path / "out"

        monkeypatch.setattr(
            "sys.argv",
            ["run_eval.py", "--config", str(cfg), "--output", str(outdir), "--dry-run"],
        )
        rc = run_eval.main()
        assert rc == 0
        assert (outdir / "t1.json").exists()
        assert (outdir / "t2.json").exists()
        assert (outdir / "summary.json").exists()

        with open(outdir / "summary.json") as f:
            summary = json.load(f)
        assert summary["config_name"] == "test"
        assert summary["num_traces"] == 2

    def test_returns_nonzero_when_no_traces(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text("name: empty\ntraces: []\n")
        outdir = tmp_path / "out"

        monkeypatch.setattr(
            "sys.argv",
            ["run_eval.py", "--config", str(cfg), "--output", str(outdir), "--dry-run"],
        )
        assert run_eval.main() == 1
