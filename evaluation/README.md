# Workstream D — Evaluation

End-to-end experimental harness for the MoM project. Defines what we measure,
how we measure it, and how results are compared across configurations.

## Primary metrics (per the action plan)

| Metric | Symbol | Where measured | Granularity |
|--------|--------|----------------|-------------|
| Time to first token | TTFT | Engine prefill exit | Per turn |
| Time between tokens | TBT  | Engine decode loop | Per turn (mean + p99) |
| Prefill recomputation time after tool return | PRT | Engine prefill, post-tool turns only | Per post-tool turn |
| Peak VRAM utilization | VRAM | `torch.cuda.max_memory_allocated()` | Per trace |
| vLLM preemption count | PRMP | Scheduler internal counter | Per trace |
| CPU↔GPU transfer time / bandwidth | XFER | Profiler events when offload enabled | Per trace |

## Secondary metrics (realism layers)

| Metric | Source | Granularity |
|--------|--------|-------------|
| AgentBench task accuracy | AgentBench harness | Per trace |
| ToolBench task accuracy  | ToolBench harness  | Per trace |

These are gated behind realistic workloads. The action plan flags them as
*realism layers, not primary evidence* — synthetic traces are the primary
benchmark because they isolate the pause-resume pattern from agent capability.

## Configuration matrix

| Config name      | KV retention | Quantization | torch.compile | LMCache |
|------------------|:------------:|:------------:|:-------------:|:-------:|
| baseline         | ✗            | ✗            | ✗             | ✗       |
| retention        | ✓            | ✗            | ✗             | ✗       |
| retention-int8   | ✓            | INT8         | ✗             | ✗       |
| retention-int4   | ✓            | INT4         | ✗             | ✗       |
| full-stack       | ✓            | INT8         | ✓             | ✗       |
| lmcache          | (LMCache)    | ✗            | ✗             | ✓       |

Each config × each turn count (5 / 10 / 25 / 50) × both models (Llama-3 8B,
Mistral 7B) defines the full experimental matrix.

## How to run

```bash
# 1. Run one config across its traces
python evaluation/run_eval.py \
    --config configs/baseline.yaml \
    --output results/baseline/

# 2. Repeat for every config in the matrix above

# 3. Emit the cross-config comparison table
python evaluation/comparison_table.py \
    --results-root results/ \
    --output results/comparison.md
```

Use `--dry-run` until Workstreams A/B/C land their engine integration. Dry-run
emits stub TraceResults (sentinel `-1.0` values) so the pipeline structure is
exercisable end-to-end before real numbers exist.

## Output schema

Every run writes:
```
results/<config_name>/
├── <trace_id_1>.json    # full TraceResult per trace
├── <trace_id_2>.json
├── ...
└── summary.json         # RunSummary aggregated across all traces
```

`summary.json` is what `comparison_table.py` consumes.

## Interfaces this workstream depends on

| From | Surface | Status |
|------|---------|--------|
| Workstream A (retention) | `src/retention/events.py` log_event() | ✅ landed (`46299ed`) |
| Workstream A (retention) | `PinManager` integration in scheduler | ✅ landed (`cfac2ba`) |
| Workstream B (quantization) | `src/quantization/` quant config + selective retention | ⏳ pending |
| Workstream C (compile + profiling) | `benchmarks/trace_generator.py` | ⏳ pending |
| Workstream C (compile + profiling) | PyTorch Profiler / Nsight glue | ⏳ pending |

## Notes

- All metrics use `time.monotonic_ns()` so timestamps align with Daksh's event
  log in `src/retention/events.py`. Do not mix in `time.time()` for latency.
- VRAM is captured at trace exit, not at peak — be careful with multi-process
  GPU sharing. Use `torch.cuda.reset_peak_memory_stats()` between traces.
- Preemption count is read from `engine.scheduler.metrics` if available; falls
  back to counting `pin_rejected_budget` events from `src/retention/events.py`.
