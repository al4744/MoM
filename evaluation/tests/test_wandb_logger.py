"""Tests for optional WandB evaluation logging."""
from __future__ import annotations

import importlib
import json
import sys

import pytest

from evaluation import run_eval
from evaluation.metrics import RunSummary, TraceResult, TurnMetrics
from evaluation.wandb_logger import WandbSettings, log_evaluation_run


class FakeTable:
    def __init__(self, columns, data):
        self.columns = columns
        self.data = data


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path, name=None):
        self.files.append((path, name))


class FakeRun:
    def __init__(self):
        self.artifacts = []

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)


class FakeWandb:
    Table = FakeTable
    Artifact = FakeArtifact

    def __init__(self):
        self.init_kwargs = None
        self.logs = []
        self.finished = False
        self.run = FakeRun()

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run

    def log(self, payload):
        self.logs.append(payload)

    def finish(self):
        self.finished = True


def _trace() -> TraceResult:
    return TraceResult(
        config_name="baseline",
        trace_id="t1",
        turns=[
            TurnMetrics(
                turn_index=0,
                ttft_ms=100.0,
                tbt_ms_mean=5.0,
                tbt_ms_p99=8.0,
                wallclock_ms=130.0,
                tool_name="search",
                tool_latency_class="fast",
                pin_hits=1,
                pin_misses=0,
                blocks_reused=4,
            )
        ],
        peak_vram_mb=1024.0,
        preemption_count=2,
        cpu_gpu_xfer_bytes=10_000_000,
        cpu_gpu_xfer_ms=50.0,
    )


def _install_fake_wandb(monkeypatch) -> FakeWandb:
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


class TestWandbLogger:
    def test_logs_summary_tables_and_artifacts(self, tmp_path, monkeypatch) -> None:
        fake = _install_fake_wandb(monkeypatch)
        cfg_path = tmp_path / "baseline.yaml"
        cfg_path.write_text("name: baseline\n")
        (tmp_path / "t1.json").write_text("{}")
        (tmp_path / "summary.json").write_text("{}")
        (tmp_path / "comparison.md").write_text("| Config |\n")
        trace = _trace()
        summary = RunSummary.from_traces([trace])

        logged = log_evaluation_run(
            settings=WandbSettings(project="mom-eval", tags=("smoke",), mode="offline"),
            cfg={"name": "baseline", "model": {"name": "mock-model"}},
            config_path=cfg_path,
            output_dir=tmp_path,
            traces=[trace],
            summary=summary,
            execution_mode="mock",
        )

        assert logged is True
        assert fake.init_kwargs["name"] == "baseline-mock"
        assert fake.init_kwargs["mode"] == "offline"
        assert set(fake.init_kwargs["tags"]) == {"smoke", "baseline", "mock"}

        summary_log = next(
            payload for payload in fake.logs if "summary/mean_ttft_ms" in payload
        )
        assert summary_log["summary/mean_ttft_ms"] == pytest.approx(100.0)
        assert summary_log["summary/mean_prefill_recomp_ms"] == 0.0
        assert summary_log["summary/p99_tbt_ms"] == 8.0
        assert summary_log["summary/mean_peak_vram_mb"] == 1024.0
        assert summary_log["summary/total_preemptions"] == 2
        assert summary_log["summary/mean_xfer_bandwidth_mb_s"] == pytest.approx(200.0)

        per_turn_log = next(payload for payload in fake.logs if "per_turn_metrics" in payload)
        per_turn_table = per_turn_log["per_turn_metrics"]
        assert per_turn_table.columns == [
            "config_name",
            "trace_id",
            "turn_index",
            "tool_name",
            "tool_latency_class",
            "ttft_ms",
            "tbt_ms_mean",
            "tbt_ms_p99",
            "wallclock_ms",
            "prefill_recomp_ms",
            "pin_hits",
            "pin_misses",
            "blocks_reused",
        ]
        assert per_turn_table.data[0][:3] == ["baseline", "t1", 0]

        assert fake.run.artifacts
        artifact_files = {name for _, name in fake.run.artifacts[0].files}
        assert "t1.json" in artifact_files
        assert "summary.json" in artifact_files
        assert "comparison.md" in artifact_files
        assert "config/baseline.yaml" in artifact_files
        assert fake.finished is True

    def test_missing_wandb_warns_and_skips(self, tmp_path, monkeypatch, capsys) -> None:
        def missing_import(name):
            if name == "wandb":
                raise ImportError("no wandb")
            return importlib.import_module(name)

        monkeypatch.setattr("evaluation.wandb_logger.importlib.import_module", missing_import)
        trace = _trace()
        summary = RunSummary.from_traces([trace])

        logged = log_evaluation_run(
            settings=WandbSettings(mode="offline"),
            cfg={"name": "baseline"},
            config_path=tmp_path / "missing.yaml",
            output_dir=tmp_path,
            traces=[trace],
            summary=summary,
            execution_mode="dry-run",
        )

        assert logged is False
        assert "wandb not installed" in capsys.readouterr().err


class TestRunEvalWandbCLI:
    def test_wandb_dry_run_does_not_crash_with_fake_wandb(self, tmp_path, monkeypatch) -> None:
        fake = _install_fake_wandb(monkeypatch)
        cfg = tmp_path / "test.yaml"
        cfg.write_text("name: test\ntraces:\n  - id: t1\n    turns: 3\n")
        outdir = tmp_path / "out"

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_eval.py",
                "--config",
                str(cfg),
                "--output",
                str(outdir),
                "--dry-run",
                "--wandb",
                "--wandb-mode",
                "offline",
            ],
        )

        assert run_eval.main() == 0
        assert (outdir / "summary.json").exists()
        assert fake.init_kwargs["name"] == "test-dry-run"
        assert any("per_turn_metrics" in payload for payload in fake.logs)
        assert fake.run.artifacts

    def test_wandb_missing_does_not_fail_run(self, tmp_path, monkeypatch, capsys) -> None:
        def missing_import(name):
            if name == "wandb":
                raise ImportError("no wandb")
            return importlib.import_module(name)

        monkeypatch.setattr("evaluation.wandb_logger.importlib.import_module", missing_import)
        cfg = tmp_path / "test.yaml"
        cfg.write_text("name: test\ntraces:\n  - id: t1\n    turns: 3\n")
        outdir = tmp_path / "out"
        monkeypatch.setattr(
            "sys.argv",
            [
                "run_eval.py",
                "--config",
                str(cfg),
                "--output",
                str(outdir),
                "--dry-run",
                "--wandb",
                "--wandb-mode",
                "offline",
            ],
        )

        assert run_eval.main() == 0
        assert "wandb not installed" in capsys.readouterr().err

    def test_without_wandb_does_not_import_wandb(self, tmp_path, monkeypatch) -> None:
        cfg = tmp_path / "test.yaml"
        cfg.write_text("name: test\ntraces:\n  - id: t1\n    turns: 3\n")
        outdir = tmp_path / "out"

        def fail_import(name):
            if name == "wandb":
                raise AssertionError("wandb should not be imported")
            return importlib.import_module(name)

        monkeypatch.setattr("evaluation.wandb_logger.importlib.import_module", fail_import)
        monkeypatch.setattr(
            "sys.argv",
            ["run_eval.py", "--config", str(cfg), "--output", str(outdir), "--dry-run"],
        )

        assert run_eval.main() == 0
        with open(outdir / "summary.json") as f:
            summary = json.load(f)
        assert summary["config_name"] == "test"
