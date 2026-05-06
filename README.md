# Mixture of Memory (MoM)

**Specialized Memory Experts for Agentic LLM Serving**

COMS E6998 - High Performance Machine Learning, Columbia University, Spring 2026

## Team

- Dakshinamoorthy A
- Alexander Ryssdal-Banoun
- Andrew Lee
- Vatsalam Krishna Jha

## Overview

Multi-turn agentic LLM workloads — where models pause to call external tools and then resume generation — suffer from expensive KV cache recomputation on every tool return. This project implements three complementary optimizations on top of vLLM to reduce that overhead:

1. **Tool-aware KV retention** — A latency predictor classifies tool calls into pin/offload/evict tiers, avoiding unnecessary reprefill.
2. **KV cache quantization** — INT8/INT4 quantization of cached KV states (not model weights), with selective retention: full precision for recent turns, quantized for older context.
3. **torch.compile** — Applied separately to prefill and decode paths, benchmarked across context lengths.

An optional extension, **MoM**, adds a routing MLP that selects among four specialized memory experts (Factual, Episodic, Semantic, Procedural) to assemble optimized prompts before they reach vLLM.

## Base Framework

This project builds on top of a vendored copy of **vLLM v0.6.4.post1** (November 2024).

This version was chosen because:
- **No KV Offloading Connector** — we build the offload/reload pipeline ourselves
- **torch.compile is experimental, not default** — we can demonstrate measurable gains
- **V0 engine architecture** — simpler to modify than the V1 engine (default in v0.8.0+)
- **INT8/INT4 KV cache quantization does not exist** in any vLLM version — our contribution is novel

The vendored vLLM source is in [`vllm/`](vllm/), forked from [`vllm-project/vllm`](https://github.com/vllm-project/vllm) at tag `v0.6.4.post1`.

## Models

- **Llama 3 8B** (primary)
- **Mistral 7B** (cross-model comparison)

## Infrastructure

- GCP VM with 2-4x A100 40GB, 128-256 GB CPU RAM, 200-300 GB SSD
- vLLM v0.6.4.post1 (vendored) + PyTorch 2.x
- WandB for experiment tracking
- PyTorch Profiler + NVIDIA Nsight Systems for profiling

## Repo Structure

```
MoM/
├── vllm/                    # Vendored vLLM v0.6.4.post1 (modifications go here)
│   ├── vllm/
│   │   ├── core/            # Scheduler, block manager (Workstream A)
│   │   ├── worker/          # CacheEngine, ModelRunner (Workstreams A+B+C)
│   │   └── ...
│   ├── csrc/                # CUDA kernels
│   ├── setup.py             # vLLM install script
│   └── ...
├── src/
│   ├── retention/           # Workstream A — KV retention policy + offload pipeline
│   ├── quantization/        # Workstream B — INT8/INT4 KV cache quantization
│   ├── compile/             # Workstream C — torch.compile integration
│   └── mom/                 # Workstream E — MoM routing + memory experts (stretch)
├── benchmarks/              # Synthetic trace generator + benchmark scripts
├── evaluation/              # Workstream D — eval scripts, metrics, comparison tables
├── configs/                 # Experiment and model configurations
├── requirements.txt         # Project dependencies (excludes vLLM, installed separately)
└── README.md
```

## Workstreams

| Workstream | Focus | Owner |
|------------|-------|-------|
| A | KV retention policy + offload pipeline | Daksh |
| B | KV cache quantization (INT8/INT4) | Vats |
| C | torch.compile + profiling infrastructure | Andrew |
| D | End-to-end evaluation | Alexander |
| E | MoM extension (stretch goal) | Alexander |

## Metrics

Primary (synthetic traces):
- Prefill recomputation time after tool return
- TTFT (time to first token)
- TBT (time between tokens)
- Peak VRAM utilization
- vLLM preemption count
- CPU↔GPU transfer time/bandwidth

Secondary (realism layers):
- AgentBench task accuracy
- ToolBench task accuracy

## Experiment Tracking

WandB is used to track inference and system benchmark runs, not model-training
curves. The evaluation runner logs only measurements already produced by the
existing evaluation pipeline.

Public WandB dashboard URL: **TODO: add the final public dashboard link after
real baseline and optimized runs are logged and the project is made public.**

When `--wandb` is enabled, `evaluation/run_eval.py` logs:
- Full YAML config as the WandB run config, plus the config file as an artifact.
- Run metadata: config name, model name, trace IDs, execution mode (`dry-run`,
  `mock`, or `real`), group, and tags.
- Aggregate summary metrics: mean TTFT, mean post-tool prefill recomputation,
  p99 TBT, mean peak VRAM, total preemptions, CPU-GPU transfer bandwidth, and
  task accuracy when present.
- Per-trace metrics for TTFT, post-tool prefill recomputation, p99 TBT, peak
  VRAM, preemption count, CPU-GPU transfer bytes/time/bandwidth, and any
  throughput or GPU-utilization fields already present in the result metadata.
- A per-turn `wandb.Table` with latency, tool, recomputation, and pin/reuse
  columns when turn-level data exists.
- Output artifacts from the evaluation output directory, including per-trace
  JSON files, `summary.json`, generated comparison markdown, CSV/JSON outputs,
  and plot files when those files already exist.

Mock smoke test without CUDA or network:

```bash
bash scripts/run_wandb_smoke_mock.sh
```

Mock runs validate logging only and are not final benchmark evidence.

Normal evaluation runs without `--wandb` do not import WandB and do not require
network access or `WANDB_API_KEY`.

Before submission, open the final dashboard URL in an incognito or logged-out
browser. Confirm the page loads without private account access and that baseline
and retention runs are visible with the intended group and tags. The team should
fill in interpretation and performance reasoning only after reviewing measured
real runs.

## VM / Final WandB Runs

Real benchmark runs belong on the GPU/vLLM VM with a compatible Python version,
preferably Python 3.11 or 3.12. Do not use local Mac/Python 3.14 runs as final
vLLM benchmark evidence.

One-time VM setup:

```bash
bash scripts/setup_vm_env.sh
source .venv/bin/activate
wandb login --relogin
export WANDB_ENTITY=al4744-col
```

Final real WandB runs:

```bash
bash scripts/run_final_wandb_eval.sh
```

The final script runs real baseline and retention evaluations, writes outputs
under `results/wandb/baseline` and `results/wandb/retention`, and logs to the
`mom-eval` project with group `final-real`. It does not use `--mock-engine` and
does not store or echo API keys.

## Setup (GCP VM)

```bash
# 1. Clone the repo
git clone https://github.com/al4744/MoM.git
cd MoM

# 2. Create the virtual environment and install dependencies
bash scripts/setup_vm_env.sh
source .venv/bin/activate

# 3. Download models
# Llama 3 8B and Mistral 7B (requires HuggingFace access)
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir models/llama3-8b
# huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir models/mistral-7b
```

## Running the evaluation pipeline

Workstream D's runner (`evaluation/run_eval.py`) drives one config across its
trace set, collects metrics, and writes per-trace JSON + an aggregated summary.
Three execution modes:

| Mode | Flag | Engine used | Needs vLLM? | Needs GPU? |
|------|------|-------------|:-----------:|:----------:|
| Dry-run | `--dry-run` | (none — sentinel `-1.0` values) | no | no |
| Mock | `--mock-engine` | `evaluation.engine_adapter.MockEngine` | no | no |
| Real | (default) | vendored `vllm.LLM` + `RetentionConfig` if enabled | yes | yes |

### Sanity checks (no GPU needed)

```bash
# Unit tests — should print "8X passed in <2s"
make test

# Smoke run — exercises the real run_eval pipeline through MockEngine
# end-to-end, then emits an ablation table. Proves wiring before GCP.
make smoke

# Explicit Python selection if your shell or VM image needs it
make PYTHON=python3 test
make PYTHON=python3 smoke
```

### Real run (GPU + vLLM required)

```bash
# 0. One-time: pull, install vendored vLLM, download model
git pull origin main
python3 -m pip install -e ./vllm
python3 -m pip install -r requirements.txt
huggingface-cli login                                   # gated model
huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir models/llama3-8b

# 1. Baseline (vanilla vLLM, no retention)
PYTHONPATH=. python3 evaluation/run_eval.py \
    --config configs/baseline.yaml \
    --output results/baseline-$(date +%Y%m%d-%H%M)/

# 2. Retention (Workstream A path active)
PYTHONPATH=. python3 evaluation/run_eval.py \
    --config configs/retention.yaml \
    --output results/retention-$(date +%Y%m%d-%H%M)/

# 3. Pairwise ablation (baseline vs retention)
make ablate \
    A=baseline-20260505-1530 \
    B=retention-20260505-1545

# 4. Cross-config comparison (every results/* subdir)
make compare
```

### Verifying retention is actually firing

Daksh's `PinManager` emits structured events to
`results/<run>/events.jsonl` via `src/retention/events.py`. After a retention
run:

```bash
RUN=results/retention-20260505-1545
ls $RUN/                                                # expect events.jsonl
grep '"event_type":"pin"'   $RUN/events.jsonl | wc -l   # > 0  → pins fired
grep '"event_type":"reuse"' $RUN/events.jsonl | wc -l   # > 0  → KV blocks reused
grep '"event_type":"pin_rejected_budget"' $RUN/events.jsonl | wc -l
                                                         # 0 ideally — non-zero
                                                         # means max_pinned_fraction
                                                         # was hit
```

If all three counts are zero, retention silently did nothing — likely a
config or engine wiring problem. Open an issue.

### Output layout

```
results/<run_name>/
├── 5turn-mixed.json     # full TraceResult per trace
├── 10turn-mixed.json
├── 25turn-mixed.json
├── 50turn-mixed.json
├── summary.json         # RunSummary (consumed by comparison_table.py)
└── events.jsonl         # Daksh's pin/reuse/expire/evict event log
```

### Make targets at a glance

| Target | What it does |
|--------|--------------|
| `make test` | Run all unit tests under `evaluation/tests/` |
| `make smoke` | Real eval pipeline through `MockEngine` (CPU-only) |
| `make eval-baseline` / `make eval-retention` | Dry-run sentinel runs |
| `make eval-all` | Dry-run every YAML under `configs/` |
| `make compare` | Cross-config markdown table from `results/*/summary.json` |
| `make ablate A=<base> B=<cand>` | Pairwise Δ + speedup table |
| `make clean` | Wipe `results/`, `__pycache__`, `.pytest_cache` |

### What's wired vs. stubbed

| Knob | YAML location | Status |
|------|---------------|:------:|
| KV retention (Workstream A) | `engine.retention.*` | ✅ wired through `LLM(retention_config=...)` |
| KV quantization (Workstream B) | `engine.quantization.kv_cache` | ✅ INT8/INT4 integrated end-to-end |
| Prefix caching | `engine.prefix_caching.enabled` | ✅ forwarded to vLLM (required when retention enabled) |
| torch.compile (Workstream C) | `engine.torch_compile.enabled` | ⏳ slot defined, env-var driver TODO |
| ~~LMCache~~ | — | ❌ descoped — pin-or-evict over PC supersedes LMCache-style tiering |

## Ablation methodology

The eval matrix is structured to attribute speedup to specific contributions
rather than to a vague "everything bundled". Every paper claim should map
to one of these deltas.

### Primary attribution — three configs, two deltas

| Config | Prefix cache | Retention | What it captures |
|---|:---:|:---:|---|
| `baseline.yaml` | ❌ | ❌ | Vanilla vLLM (reference) |
| `prefix_cache_only.yaml` | ✅ | ❌ | vLLM's built-in PC contribution |
| `retention.yaml` | ✅ | ✅ | PC + our pin-or-evict on top |

| Delta | Attribution |
|---|---|
| `baseline → prefix_cache_only` | Speedup from vLLM's existing prefix caching |
| **`prefix_cache_only → retention`** | **Speedup specifically attributable to our pin-or-evict policy** |
| `baseline → retention` | Total combined speedup (the paper's headline) |

The fourth combination (`PC=off, retention=on`) does not exist by design:
Daksh's `LLMEngine.__init__` raises `ValueError` if retention is enabled
without prefix caching, because pin-or-evict relies on the prefix-cache hash
lookup to revive held blocks. We document this constraint rather than work
around it.

### Retention component ablations — three configs

These hold retention enabled and toggle individual sub-mechanisms.

| Config | Toggle | What it isolates |
|---|---|---|
| `retention_no_ema.yaml` | `ttl.use_ema = false` | Predictor learning: with EMA off, TTL is a constant `default_ttl × safety_factor` |
| `retention_no_per_tool.yaml` | `ttl.use_per_tool_ema = false` | Per-tool tracking: single global EMA across all tools |
| `retention_no_pin.yaml` | `pin_manager.max_pinned_fraction = 0.001` | Pin budget: effectively zero, every pin rejected |

| Delta | Attribution |
|---|---|
| `retention_no_ema → retention` | The TTL-learning predictor's contribution |
| `retention_no_per_tool → retention` | Per-tool latency tracking's contribution |
| `retention_no_pin → retention` | The pin mechanism itself (vs. PC alone) |

`retention_no_pin` doubles as an internal consistency check: it should
produce numbers essentially identical to `prefix_cache_only` because the
PinManager is constructed but does nothing observable.

### Quantization ablations — already in the matrix

| Config | What it captures |
|---|---|
| `retention_int8.yaml` | INT8 KV quant (~50% KV memory) |
| `retention_int4.yaml` | INT4 KV quant (~25% KV memory, more lossy) |

| Delta | Attribution |
|---|---|
| `retention → retention_int8` | INT8's memory savings, accuracy cost |
| `retention → retention_int4` | INT4's memory savings, accuracy cost |

### Accuracy attribution

Latency-only is half the story. `scripts/run_accuracy_eval.py` runs
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
against each engine config (default task: MMLU). Paired with the latency
tables, this defends the quantization claims:

```bash
PYTHONPATH=. python scripts/run_accuracy_eval.py \
    --config configs/retention_int8.yaml \
    --output results/<run>/retention_int8/ \
    --tasks mmlu \
    --limit 50
```

Accuracy results land in `<output>/accuracy.json` and `summary.json` is
updated in place with `mean_task_accuracy`, picked up by the comparison
and ablation tables automatically.

## Quick Start (full project)

```bash
# Generate synthetic traces (Workstream C — not yet implemented)
python3 benchmarks/trace_generator.py --turns 50 --output traces/

# Run baseline benchmark (uses evaluation/run_eval.py — see above)
make eval-baseline
```

## AI Tool Use

TODO: Insert the required disclosure block from the HPML AI Use Policy before final submission.

## References

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention-based LLM serving (v0.6.4.post1)
- [LMCache](https://github.com/LMCache/LMCache) — KV cache management
- Shazeer et al. 2017 — Mixture of Experts
- MemGPT, Scissorhands, H₂O — Memory management for LLMs
- KIVI, KVQuant — KV cache quantization research
