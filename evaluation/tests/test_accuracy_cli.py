"""Tests for scripts/run_accuracy_eval.py and scripts/compare_accuracy.py.

These don't exercise lm-eval-harness itself (which requires a real model
load + GPU). They cover:
  * the task-suite shortcut logic — TASK_SUITES dict and parse_args wiring
  * accuracy.json → comparison-table extraction in compare_accuracy.py

For an actual end-to-end accuracy check, run scripts/run_accuracy_eval.py
on a real config — these unit tests are the local-CI safety net.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make scripts/ importable for the parse_args helpers.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# run_accuracy_eval.py — TASK_SUITES + parse_args
# ---------------------------------------------------------------------------

class TestTaskSuites:
    def test_task_suites_contains_expected_keys(self) -> None:
        from run_accuracy_eval import TASK_SUITES
        assert set(TASK_SUITES.keys()) == {"mmlu", "reasoning", "agentic"}

    def test_mmlu_suite_is_just_mmlu(self) -> None:
        from run_accuracy_eval import TASK_SUITES
        assert TASK_SUITES["mmlu"] == "mmlu"

    def test_reasoning_suite_includes_gsm8k_and_bbh(self) -> None:
        from run_accuracy_eval import TASK_SUITES
        assert "gsm8k" in TASK_SUITES["reasoning"]
        assert "bbh" in TASK_SUITES["reasoning"]

    def test_agentic_suite_combines_others(self) -> None:
        from run_accuracy_eval import TASK_SUITES
        agentic = TASK_SUITES["agentic"]
        for task in ("mmlu", "gsm8k", "bbh"):
            assert task in agentic


class TestAccuracyParseArgs:
    def test_default_falls_back_to_mmlu(self, monkeypatch) -> None:
        import run_accuracy_eval
        monkeypatch.setattr(sys, "argv", [
            "run_accuracy_eval", "--config", "x.yaml", "--output", "out/"
        ])
        args = run_accuracy_eval.parse_args()
        assert args.tasks == "mmlu"
        assert args._tasks_origin == "default-mmlu"

    def test_explicit_tasks_passes_through(self, monkeypatch) -> None:
        import run_accuracy_eval
        monkeypatch.setattr(sys, "argv", [
            "run_accuracy_eval", "--config", "x.yaml", "--output", "out/",
            "--tasks", "hellaswag,arc_easy"
        ])
        args = run_accuracy_eval.parse_args()
        assert args.tasks == "hellaswag,arc_easy"
        assert args._tasks_origin == "explicit"

    def test_task_suite_resolves_to_task_string(self, monkeypatch) -> None:
        import run_accuracy_eval
        monkeypatch.setattr(sys, "argv", [
            "run_accuracy_eval", "--config", "x.yaml", "--output", "out/",
            "--task-suite", "reasoning"
        ])
        args = run_accuracy_eval.parse_args()
        assert "gsm8k" in args.tasks
        assert "bbh" in args.tasks
        assert args._tasks_origin == "suite:reasoning"

    def test_tasks_and_task_suite_mutually_exclusive(self, monkeypatch) -> None:
        import run_accuracy_eval
        monkeypatch.setattr(sys, "argv", [
            "run_accuracy_eval", "--config", "x.yaml", "--output", "out/",
            "--tasks", "mmlu", "--task-suite", "reasoning",
        ])
        # argparse exits with SystemExit(2) on mutually-exclusive violation.
        with pytest.raises(SystemExit):
            run_accuracy_eval.parse_args()


# ---------------------------------------------------------------------------
# compare_accuracy.py — markdown table generation
# ---------------------------------------------------------------------------

def _write_accuracy_blob(dir_path: Path, config_name: str,
                         per_task: dict[str, float]) -> None:
    """Helper to drop an accuracy.json that compare_accuracy.py expects."""
    dir_path.mkdir(parents=True, exist_ok=True)
    blob = {
        "config_name": config_name,
        "tasks": ",".join(per_task.keys()),
        "limit": 50,
        "batch_size": 16,
        "results": {
            task: {"acc,none": acc} for task, acc in per_task.items()
        },
    }
    with open(dir_path / "accuracy.json", "w") as f:
        json.dump(blob, f)


class TestCompareAccuracy:
    def test_extract_per_task_uses_acc_none(self) -> None:
        import compare_accuracy
        blob = {"results": {"mmlu": {"acc,none": 0.51}}}
        out = compare_accuracy._extract_per_task(blob)
        assert out == {"mmlu": 0.51}

    def test_extract_per_task_falls_back_to_acc_norm(self) -> None:
        import compare_accuracy
        blob = {"results": {"hellaswag": {"acc_norm,none": 0.78}}}
        out = compare_accuracy._extract_per_task(blob)
        assert out == {"hellaswag": 0.78}

    def test_extract_per_task_skips_non_numeric(self) -> None:
        import compare_accuracy
        blob = {"results": {"mmlu": {"acc,none": "not_a_number"}}}
        out = compare_accuracy._extract_per_task(blob)
        assert out == {}

    def test_mean_acc_of_empty_returns_zero(self) -> None:
        import compare_accuracy
        assert compare_accuracy._mean_acc({}) == 0.0

    def test_mean_acc_of_nonempty(self) -> None:
        import compare_accuracy
        assert compare_accuracy._mean_acc(
            {"mmlu": 0.5, "gsm8k": 0.7}
        ) == pytest.approx(0.6)

    def test_full_run_against_two_dirs(self, tmp_path, capsys, monkeypatch) -> None:
        """End-to-end: write 2 accuracy.json files, invoke compare_accuracy.main."""
        import compare_accuracy
        d1 = tmp_path / "baseline"
        d2 = tmp_path / "retention"
        _write_accuracy_blob(d1, "baseline", {"mmlu": 0.50, "gsm8k": 0.40})
        _write_accuracy_blob(d2, "retention", {"mmlu": 0.49, "gsm8k": 0.41})

        monkeypatch.setattr(sys, "argv", [
            "compare_accuracy", str(d1), str(d2),
        ])
        rc = compare_accuracy.main()
        captured = capsys.readouterr()
        assert rc == 0
        # Markdown header
        assert "| config |" in captured.out
        assert "mmlu" in captured.out
        assert "gsm8k" in captured.out
        # Both rows present.
        assert "baseline" in captured.out
        assert "retention" in captured.out
        # Headline section.
        assert "Headline" in captured.out

    def test_warns_when_accuracy_drops_more_than_two_pp(self, tmp_path,
                                                        capsys, monkeypatch) -> None:
        """retention loses 3pp → output should include the ⚠ marker."""
        import compare_accuracy
        d1 = tmp_path / "baseline"
        d2 = tmp_path / "retention_int4"
        _write_accuracy_blob(d1, "baseline", {"mmlu": 0.60, "gsm8k": 0.50})
        # int4: -3 pp on each task → -3 pp on the mean.
        _write_accuracy_blob(d2, "retention_int4",
                             {"mmlu": 0.57, "gsm8k": 0.47})

        monkeypatch.setattr(sys, "argv", [
            "compare_accuracy", str(d1), str(d2),
        ])
        compare_accuracy.main()
        captured = capsys.readouterr()
        assert "⚠" in captured.out

    def test_missing_accuracy_json_exits(self, tmp_path, monkeypatch) -> None:
        import compare_accuracy
        empty = tmp_path / "empty"
        empty.mkdir()
        monkeypatch.setattr(sys, "argv", [
            "compare_accuracy", str(empty),
        ])
        with pytest.raises(SystemExit):
            compare_accuracy.main()
