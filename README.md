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

## Setup (GCP VM)

```bash
# 1. Clone the repo
git clone https://github.com/al4744/MoM.git
cd MoM

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install the vendored vLLM in editable mode
cd vllm
pip install -e .
cd ..

# 4. Install project dependencies
pip install -r requirements.txt

# 5. Download models
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
```

### Real run (GPU + vLLM required)

```bash
# 0. One-time: pull, install vendored vLLM, download model
git pull origin main
cd vllm && pip install -e . && cd ..
pip install -r requirements.txt
huggingface-cli login                                   # gated model
huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir models/llama3-8b

# 1. Baseline (vanilla vLLM, no retention)
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/baseline.yaml \
    --output results/baseline-$(date +%Y%m%d-%H%M)/

# 2. Retention (Workstream A path active)
PYTHONPATH=. python evaluation/run_eval.py \
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
| KV quantization (Workstream B) | `engine.quantization.kv_cache` | ⏳ slot defined, engine support TODO |
| torch.compile (Workstream C) | `engine.torch_compile.enabled` | ⏳ slot defined, env-var driver TODO |
| LMCache (Workstream A comparison) | `engine.lmcache.enabled` | ⏳ slot defined, integration TODO |

## Quick Start (full project)

```bash
# Generate synthetic traces (Workstream C — not yet implemented)
python benchmarks/trace_generator.py --turns 50 --output traces/

# Run baseline benchmark (uses evaluation/run_eval.py — see above)
make eval-baseline
```

## References

- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention-based LLM serving (v0.6.4.post1)
- [LMCache](https://github.com/LMCache/LMCache) — KV cache management
- Shazeer et al. 2017 — Mixture of Experts
- MemGPT, Scissorhands, H₂O — Memory management for LLMs
- KIVI, KVQuant — KV cache quantization research
