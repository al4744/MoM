# Workstream B Deliverable (KV Cache Quantization)

This document captures what is implemented for Workstream B and how to run it.

## Completed implementation

- INT8/INT4 KV cache quantization module:
  - `src/quantization/config.py`
  - `src/quantization/types.py`
  - `src/quantization/quantizer.py`
  - `src/quantization/policy.py`
  - `src/quantization/tests/*`
- vLLM integration hooks for quantized swap-out/swap-in:
  - `vllm/vllm/worker/cache_engine.py`
  - `vllm/vllm/engine/llm_engine.py`
  - `vllm/vllm/config.py`
- Quantization experiment configs:
  - `configs/retention_int8.yaml`
  - `configs/retention_int4.yaml`
- Evaluation/reporting updates for quantized KV metrics:
  - `evaluation/run_eval.py`
  - `evaluation/metrics.py`
  - `evaluation/comparison_table.py`
  - tests in `evaluation/tests/*`
- Makefile pipelines:
  - `make eval-all-real`
  - `make compare-real`
  - `make ablate-real A=... B=...`

## Verification status

- Full test suite passes:
  - `make test`
  - Result: `104 passed`

## Runbook

From repo root:

```bash
make test
make eval-all-real
make compare-real
make ablate-real A=baseline B=retention_int8
make ablate-real A=baseline B=retention_int4
```

## Output artifacts

- `results_real/comparison.md`
- `results_real/ablate-baseline-vs-retention_int8.md`
- `results_real/ablate-baseline-vs-retention_int4.md`
- Per-config JSON summaries:
  - `results_real/baseline/summary.json`
  - `results_real/retention/summary.json`
  - `results_real/retention_int8/summary.json`
  - `results_real/retention_int4/summary.json`

## Current measurable Workstream B outcome

- INT8 path shows `mean_kv_compression_ratio = 0.5`
- INT4 path shows `mean_kv_compression_ratio = 0.25`

These are available in:

- `results_real/retention_int8/summary.json`
- `results_real/retention_int4/summary.json`

## Remaining non-code project step (team benchmarking)

For final paper-quality reporting, plug in your team's benchmark workloads
(AgentBench/ToolBench and integrated A+B+C runs) to replace synthetic run-path
metrics with production measurements.
