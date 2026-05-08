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


class TestBuildVllmKwargs:
    """Regression: vLLM 0.6.4 SchedulerConfig._verify_args fails when
    max_num_seqs is None (lm-eval's vllm_causallms wrapper passes it
    through without defaulting). Without an explicit default in our
    kwargs builder, the comparison `max_num_batched_tokens < max_num_seqs`
    raises TypeError. This test pins the fix in place.
    """

    def test_max_num_seqs_always_present(self) -> None:
        import run_accuracy_eval
        cfg = {"model": {"name": "meta-llama/Meta-Llama-3-8B"}}
        kwargs = run_accuracy_eval._build_vllm_kwargs(cfg)
        assert "max_num_seqs" in kwargs, (
            "max_num_seqs missing from vLLM kwargs — lm-eval call will crash "
            "in vLLM 0.6.4 SchedulerConfig._verify_args."
        )
        assert isinstance(kwargs["max_num_seqs"], int)
        assert kwargs["max_num_seqs"] > 0

    def test_max_num_seqs_default_is_256(self) -> None:
        import run_accuracy_eval
        cfg = {"model": {"name": "x"}}
        kwargs = run_accuracy_eval._build_vllm_kwargs(cfg)
        assert kwargs["max_num_seqs"] == 256

    def test_max_num_seqs_overridable_from_yaml(self) -> None:
        import run_accuracy_eval
        cfg = {"model": {"name": "x", "max_num_seqs": 64}}
        kwargs = run_accuracy_eval._build_vllm_kwargs(cfg)
        assert kwargs["max_num_seqs"] == 64

    def test_existing_kwargs_still_present(self) -> None:
        """Don't accidentally drop pretrained/dtype/etc."""
        import run_accuracy_eval
        cfg = {
            "model": {
                "name": "meta-llama/Meta-Llama-3-8B",
                "dtype": "float16",
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.85,
            }
        }
        kwargs = run_accuracy_eval._build_vllm_kwargs(cfg)
        assert kwargs["pretrained"] == "meta-llama/Meta-Llama-3-8B"
        assert kwargs["dtype"] == "float16"
        assert kwargs["max_model_len"] == 4096
        assert kwargs["gpu_memory_utilization"] == 0.85


class TestExtractMeanAccuracy:
    """Regression: _extract_mean_accuracy returned None for gsm8k+bbh runs
    because it only matched acc/acc_norm metric heads, not exact_match.
    First production accuracy run logged 'no acc/acc_norm metrics found
    in results' for both baseline and retention configs."""

    def test_handles_acc_metric(self) -> None:
        import run_accuracy_eval
        results = {"results": {"mmlu": {"acc,none": 0.50}}}
        assert run_accuracy_eval._extract_mean_accuracy(results) == 0.50

    def test_handles_exact_match_for_gsm8k(self) -> None:
        import run_accuracy_eval
        results = {"results": {"gsm8k": {
            "exact_match,strict-match": 0.40,
            "exact_match,flexible-extract": 0.45,
        }}}
        # Should pick exact_match (strict-match first), giving 0.40
        assert run_accuracy_eval._extract_mean_accuracy(results) == 0.40

    def test_no_double_counting_within_task(self) -> None:
        """A single task with both acc and exact_match should contribute
        ONE value to the mean, not two."""
        import run_accuracy_eval
        results = {"results": {"weird_task": {
            "acc,none": 0.50,
            "exact_match,strict-match": 0.30,
        }}}
        # Picks acc (higher priority), value=0.50, mean=0.50 (not avg of both)
        assert run_accuracy_eval._extract_mean_accuracy(results) == 0.50

    def test_averages_across_tasks(self) -> None:
        import run_accuracy_eval
        results = {"results": {
            "mmlu":  {"acc,none": 0.50},
            "gsm8k": {"exact_match,strict-match": 0.30},
        }}
        # Mean of 0.50 and 0.30
        assert run_accuracy_eval._extract_mean_accuracy(results) == pytest.approx(0.40)

    def test_returns_none_when_no_metrics(self) -> None:
        import run_accuracy_eval
        results = {"results": {"task": {"some_other_metric": 0.5}}}
        assert run_accuracy_eval._extract_mean_accuracy(results) is None

    def test_returns_none_for_empty_results(self) -> None:
        import run_accuracy_eval
        assert run_accuracy_eval._extract_mean_accuracy({"results": {}}) is None
        assert run_accuracy_eval._extract_mean_accuracy({}) is None


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

    def test_extract_per_task_handles_gsm8k_exact_match(self) -> None:
        """Regression: gsm8k uses exact_match,strict-match instead of acc.
        First production accuracy run logged 'no acc/acc_norm metrics found'
        because we only checked acc/acc_norm."""
        import compare_accuracy
        blob = {"results": {"gsm8k": {
            "exact_match,strict-match": 0.36,
            "exact_match,flexible-extract": 0.42,
        }}}
        out = compare_accuracy._extract_per_task(blob)
        # Should pick strict-match (higher priority in our metric order)
        assert out == {"gsm8k": 0.36}

    def test_extract_per_task_handles_bbh_mixed_metrics(self) -> None:
        """BBH subtasks vary; some use acc, some exact_match. Make sure we
        pick the right one per-task without double-counting."""
        import compare_accuracy
        blob = {"results": {
            "bbh_logical_deduction":  {"acc,none": 0.45},
            "bbh_word_sorting":       {"exact_match,flexible-extract": 0.30},
        }}
        out = compare_accuracy._extract_per_task(blob)
        assert out == {"bbh_logical_deduction": 0.45,
                       "bbh_word_sorting": 0.30}

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
