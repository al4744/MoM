# Mixture of Memory (MoM)

**Specialized Memory Experts for Agentic LLM Serving**

COMS E6998 - High Performance Machine Learning, Columbia University, Spring 2026

## Team

- Dakshinamoorthy A
- Alexander Ryssdal-Banoun
- Andrew Lee
- Vatsalam Krishna Jha

## Overview

Multi-turn agentic LLM workloads - where models pause to call external tools and then resume generation - suffer from expensive KV cache recomputation on every tool return. This project implements three complementary optimizations on top of vLLM to reduce that overhead:

1. **Tool-aware KV retention** - A latency predictor classifies tool calls into pin/offload/evict tiers using vLLM's KV Offloading Connector, avoiding unnecessary reprefill.
2. **KV cache quantization** - INT8/INT4 quantization of cached KV states (not model weights), with selective retention: full precision for recent turns, quantized for older context.
3. **torch.compile** - Applied separately to prefill and decode paths, benchmarked across context lengths.

An optional extension, **MoM**, adds a routing MLP that selects among four specialized memory experts (Factual, Episodic, Semantic, Procedural) to assemble optimized prompts before they reach vLLM.

## Models

- **Llama 3 8B** (primary)
- **Mistral 7B** (cross-model comparison)

## Infrastructure

- GCP VM with 2-4x A100 40GB, 128-256 GB CPU RAM, 200-300 GB SSD
- vLLM (latest) + PyTorch 2.x
- WandB for experiment tracking
- PyTorch Profiler + NVIDIA Nsight Systems for profiling

## Repo Structure

```
MoM/
├── src/
│   ├── retention/       # Workstream A - KV retention policy + offload pipeline
│   ├── quantization/    # Workstream B - INT8/INT4 KV cache quantization
│   ├── compile/         # Workstream C - torch.compile integration
│   └── mom/             # Workstream E - MoM routing + memory experts (stretch)
├── benchmarks/          # Synthetic trace generator + benchmark scripts
├── configs/             # Experiment and model configurations
├── evaluation/          # Workstream D - eval scripts, metrics, comparison tables
├── profiling/           # PyTorch Profiler traces, Nsight timelines
├── scripts/             # Utility scripts (setup, data download, etc.)
├── requirements.txt
└── README.md
```

## Workstreams

| Workstream | Focus | Owner |
|------------|-------|-------|
| A | KV retention policy via Offloading Connector | Daksh |
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
- CPU-GPU transfer time/bandwidth

Secondary (realism layers):
- AgentBench task accuracy
- ToolBench task accuracy

## Quick Start

```bash
# Clone
git clone https://github.com/al4744/MoM.git
cd MoM

# Install dependencies
pip install -r requirements.txt

# Generate synthetic traces
python benchmarks/trace_generator.py --turns 50 --output traces/

# Run baseline benchmark
python benchmarks/run_benchmark.py --config configs/baseline.yaml
```

## References

- [vLLM](https://github.com/vllm-project/vllm) - PagedAttention-based LLM serving
- [LMCache](https://github.com/LMCache/LMCache) - KV cache management
- Shazeer et al. 2017 - Mixture of Experts
- MemGPT, Scissorhands, H2O - Memory management for LLMs
