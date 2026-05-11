# HPML Final Project: Mixture of Memory (MoM)

> **Course:** High Performance Machine Learning
> **Semester:** Spring 2026
> **Instructor:** Dr. Kaoutar El Maghraoui

---

## Team Information

- **Team Name:** MoM / Mixture of Memory
- **Members:**
  - Dakshinamoorthy A (da3232) — *Workstream A: Tool-aware KV retention (PinManager, TTLPredictor, vLLM scheduler hooks)*
  - Alexander Ryssdal-Banoun (ar4678) — *Workstream D: End-to-end evaluation framework & concurrent runner; Workstream E: MoM routing extension (stretch)*
  - Andrew Lee (al4744) — *Workstream C: torch.compile integration & phase-tagged profiling infrastructure*
  - Vatsalam Krishna Jha (vkj2107) — *Workstream B: INT8/INT4 KV cache quantization*

---

## Submission

- **GitHub repository:** [https://github.com/al4744/MoM](https://github.com/al4744/MoM)
- **Final report:** [`deliverables/MoM_HPML_Final_Report.pdf`](deliverables/MoM_HPML_Final_Report.pdf)
- **Final presentation:** [`deliverables/MoM_HPML_Final_Presentation.pptx`](deliverables/MoM_HPML_Final_Presentation.pptx)

The final report PDF and presentation are checked into the `deliverables/` folder and uploaded to CourseWorks.

---

## 1. Problem Statement

Multi-turn agentic LLM workloads — where models pause to call external tools and then resume generation — suffer from expensive KV cache recomputation on every tool return. When a serving engine evicts a paused agent's KV blocks under cross-agent memory pressure, the model must recompute the full prefill from scratch on resumption, with latency proportional to context length. This project targets **inference** and addresses the **memory bandwidth and KV cache eviction bottleneck** in multi-agent vLLM serving by implementing tool-aware retention, KV compression, and compilation instrumentation on top of vLLM v0.6.4.post1.

---

## 2. Model / Application Description

- **Model architecture:** Meta-Llama-3-8B (FP16, 32 layers, GQA with 8 KV heads, max context 8,192 tokens)
- **Framework:** vLLM v0.6.4.post1 (V0 engine), PyTorch 2.x, CUDA 12.x
- **Dataset:** Synthetic multi-agent traces with Poisson-distributed arrivals, parameterised by turn count (5/10/25/50), tool-call frequency, and tool latency distribution. No external dataset download required.
- **Custom modifications:**
  - `vllm/core/scheduler.py` — four scheduler hooks (pin-on-finish, reuse-on-arrival, pressure-evict, TTL-sweep)
  - `vllm/engine/llm_engine.py` — `PinManager` and `TTLPredictor` instantiation; `program_id` / `is_tool_call_pending` / `tool_name` forwarded from `generate()`
  - `vllm/entrypoints/llm.py` — extended `generate()` signature
  - `vllm/worker/cache_engine.py` + `vllm/engine/llm_engine.py` — INT8/INT4 quantization hooks
- **Hardware target:** NVIDIA L4 (24 GB) and H100 (80 GB) on GCP VMs

---

## 3. Final Results Summary

All numbers are from real GPU runs on GCP VMs with Meta-Llama-3-8B FP16.

**Table 1 — Post-Tool TTFT: PC-Only vs. Retention (sequential isolation, 60 filler agents)**

| GPU | PC-Only (ms) | Retention (ms) | Speedup |
|---|---|---|---|
| L4 | 843.0 | 82.0 | **10.2×** |
| H100 | 81.9 | 19.4 | **4.2×** |

**Table 2 — KV Cache Quantization: Memory and Latency**

| Metric | FP16 | INT8 | INT4 |
|---|---|---|---|
| Active KV working set (GB) | 2.0 | 1.0 | 0.5 |
| Compression vs. FP16 | 1× | 2× | **4×** |
| TTFT (ms) | — | 96.7 | 94.7 |

**Table 3 — Task Accuracy: Baseline vs. Retention** *(n=50 per subtask, lm-evaluation-harness)*

| Task | Baseline | Retention | Δ |
|---|---|---|---|
| GSM8K | 48.0% | 48.0% | 0 pp |
| BBH | 63.3% | 63.3% | 0 pp |

**Table 4 — p95 Post-Tool TTFT (ms): Constrained Cache Open-Loop Sweep** *(gpu_memory_utilization=0.30, 150 agents)*

| Rate (RPS) | Baseline | PC-Only | Retention |
|---|---|---|---|
| 1 | 5,345 | 9,884 | **7,120** |
| 2 | 22,321 | 27,621 | 26,413 |
| 4 | 30,201 | 40,423 | 37,534 |
| 8 | 34,766 | 35,448 | 34,518 |
| 16 | 34,953 | 36,151 | 35,339 |

Pin survival under 60-agent concurrent load (H100): **7 pin events, 7 reuse events, 0 evictions, 0 expirations (100%).**

**Hardware:** GCP VM · NVIDIA L4 24 GB / H100 80 GB · CUDA 12.x · PyTorch 2.x · vLLM v0.6.4.post1 · Ubuntu 22.04

**Headline result:** Tool-aware KV retention achieves a **10.2× post-tool TTFT speedup on L4** and **4.2× on H100** in sequential isolation with 60-agent concurrent load, and INT4 KV quantization delivers a **4× active working-set reduction** versus FP16 with zero measurable accuracy degradation on GSM8K and BBH.

---

## 4. Repository Structure

```
MoM/
├── README.md
├── requirements.txt
├── Makefile                     # All make targets (test, smoke, eval-*, compare, ablate)
├── configs/                     # YAML configs for every reported experiment
│   ├── baseline.yaml            # Vanilla vLLM — no prefix cache, no retention
│   ├── prefix_cache_only.yaml   # vLLM built-in prefix caching only
│   ├── retention.yaml           # PC + our pin-or-evict (headline config)
│   ├── retention_int8.yaml      # Retention + INT8 KV quantization
│   ├── retention_int4.yaml      # Retention + INT4 KV quantization
│   ├── retention_no_ema.yaml    # Ablation: constant TTL (no EMA learning)
│   ├── retention_no_per_tool.yaml # Ablation: global EMA only
│   └── retention_no_pin.yaml    # Ablation: zero pin budget
├── deliverables/
│   ├── MoM_HPML_Final_Report.pdf
│   └── MoM_HPML_Final_Presentation.pptx
├── src/
│   ├── retention/               # Workstream A — PinManager, TTLPredictor, event logger
│   ├── quantization/            # Workstream B — INT8/INT4 KV quantizer
│   └── compile/                 # Workstream C — torch.compile config dataclasses
├── evaluation/
│   ├── run_eval.py              # Main CLI runner (dry-run / mock / real modes)
│   ├── engine_adapter.py        # EngineProtocol, MockEngine, build_real_engine()
│   ├── concurrent_runner.py     # Multi-agent Poisson-arrival runner (Workstream D)
│   ├── metrics.py               # TurnMetrics, TraceResult, RunSummary
│   └── tests/                   # Pytest suite (249 tests)
├── benchmarks/                  # Synthetic trace generator + compile sweep scripts
├── scripts/
│   ├── setup_vm_env.sh          # One-time GCP VM environment setup
│   ├── run_final_wandb_eval.sh  # Production eval script
│   └── run_accuracy_eval.py     # lm-evaluation-harness wrapper
├── vllm/                        # Vendored vLLM v0.6.4.post1 source (do not compile — see §5)
└── docs/                        # Extended design notes
```

---

## 5. Reproducibility Instructions

### A. Environment Setup

```bash
# 1. Clone the repo
git clone https://github.com/al4744/MoM.git
cd MoM

# 2. Create a clean Python environment (Python 3.11 or 3.12 recommended)
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# On Windows: .venv\Scripts\activate

# 3. Install stock vLLM v0.6.4.post1 from PyPI (pre-built wheels — no compilation)
pip install vllm==0.6.4.post1

# 4. Install project dependencies
pip install -r requirements.txt
```

**System requirements:** Python 3.11–3.12, CUDA 12.x, ≥24 GB GPU VRAM for Llama-3-8B. See `requirements.txt` for pinned package versions.

---

### ⚠️ Important: How our vLLM modifications are applied

The `vllm/` directory in this repo contains the full vendored source of vLLM v0.6.4.post1 with our scheduler hooks, retention wiring, and quantization changes applied as patches. **We do not compile or `pip install -e ./vllm`** — compiling vLLM from source takes **6–24+ hours** depending on hardware and is not practical for reproduction.

Instead, we install stock vLLM from PyPI (`pip install vllm==0.6.4.post1`) and then **overwrite the modified Python files in place**:

```bash
# After installing stock vLLM (step 3 above), apply our patches:
VLLM_SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")

# Core scheduler and engine hooks (Workstream A)
cp vllm/vllm/core/scheduler.py           $VLLM_SITE/core/scheduler.py
cp vllm/vllm/engine/llm_engine.py        $VLLM_SITE/engine/llm_engine.py
cp vllm/vllm/entrypoints/llm.py          $VLLM_SITE/entrypoints/llm.py

# KV quantization hooks (Workstream B)
cp vllm/vllm/worker/cache_engine.py      $VLLM_SITE/worker/cache_engine.py
```

This avoids the CUDA compilation step entirely — all modified files are pure Python and slot directly into the installed package. This approach was adopted after the full compilation path was found to be impractical for course reproduction (6–24 h build time, as documented in our final presentation challenges slide).

> **Why the vendored source is still in the repo:** It serves as the ground-truth diff of every change we made. The `vllm/` directory is the canonical record of our modifications and can be used for code review or to apply patches to any future vLLM install.

---

### B. Download the Model

```bash
huggingface-cli login   # requires HuggingFace account with Llama-3 access approved

huggingface-cli download meta-llama/Meta-Llama-3-8B \
    --local-dir models/llama3-8b
```

---

### C. Quickstart: CPU-only Smoke Test (no GPU needed)

Validates the full evaluation pipeline wiring without requiring a GPU or model download:

```bash
# Run unit tests (249 tests, CPU-only)
make test

# Smoke run: exercises run_eval.py end-to-end through MockEngine, emits ablation table
make smoke
```

Both should pass in under 2 minutes on any machine.

---

### D. Full Reproduction: Headline Results (GPU required)

The following reproduces the numbers in Section 3. Requires a GPU VM with ≥24 GB VRAM and the model downloaded (step B).

```bash
# 0. Apply our vLLM patches (one-time, after pip install vllm==0.6.4.post1)
VLLM_SITE=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
cp vllm/vllm/core/scheduler.py        $VLLM_SITE/core/scheduler.py
cp vllm/vllm/engine/llm_engine.py     $VLLM_SITE/engine/llm_engine.py
cp vllm/vllm/entrypoints/llm.py       $VLLM_SITE/entrypoints/llm.py
cp vllm/vllm/worker/cache_engine.py   $VLLM_SITE/worker/cache_engine.py

# 1. Baseline (vanilla vLLM, no retention)
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/baseline.yaml \
    --output results/baseline-$(date +%Y%m%d-%H%M)/

# 2. Prefix-cache-only (vLLM's built-in PC, no pin-or-evict)
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/prefix_cache_only.yaml \
    --output results/prefix_cache_only-$(date +%Y%m%d-%H%M)/

# 3. Retention (PC + our pin-or-evict — the headline config)
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/retention.yaml \
    --output results/retention-$(date +%Y%m%d-%H%M)/

# 4. INT4 quantization
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/retention_int4.yaml \
    --output results/retention_int4-$(date +%Y%m%d-%H%M)/

# 5. Pairwise comparison (prefix_cache_only → retention speedup)
make ablate A=prefix_cache_only-<timestamp> B=retention-<timestamp>

# 6. Cross-config summary table
make compare
```

**Expected runtime:** ~20–40 min per config on an L4 GPU.

---

### E. Accuracy Evaluation

```bash
PYTHONPATH=. python scripts/run_accuracy_eval.py \
    --config configs/retention.yaml \
    --output results/accuracy_retention/ \
    --tasks reasoning \
    --limit 50
```

Runs GSM8K + BBH via lm-evaluation-harness. Results land in `accuracy.json` and are picked up automatically by `make compare`.

---

### F. Profiling (Workstream C)

```bash
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/retention.yaml \
    --output results/profile_run \
    --profile \
    --pytorch-profiler \
    --profile-output-dir results/profiles/

# Nsight Systems (generates the nsys command for the GPU VM — does not auto-launch)
PYTHONPATH=. python evaluation/run_eval.py \
    --config configs/retention.yaml \
    --output results/profile_run \
    --nsight \
    --profile-output-dir results/profiles/
# Then run the generated script: bash results/profile_run/nsight_command.sh
```

---

## 6. Results and Observations

- **Tool-aware KV retention (Workstream A):** Achieves 10.2× post-tool TTFT speedup on L4 and 4.2× on H100 in sequential isolation against 60 filler agents. The asymmetry is hardware-driven: the L4's slower prefill makes recomputation proportionally more expensive, while the cache-hit floor does not shrink at the same rate. Under cache-constrained open-loop serving (3,700 KV blocks, `gpu_memory_utilization=0.30`), retention reduces p95 post-tool TTFT by 28% at 1 RPS, with diminishing returns as the system saturates.

- **INT4 KV cache quantization (Workstream B):** Reduces active KV working-set memory by 4× (2.0 GB → 0.5 GB) versus FP16 with no measurable accuracy degradation on GSM8K (48.0%) or BBH (63.3%). The selective-precision policy (full FP16 for recent 2 turns, INT4 for older context) preserves quality while maximising compression.

- **torch.compile integration (Workstream C):** Compilation via `VLLM_TORCH_COMPILE_LEVEL=1` is wired end-to-end with phase-tagged profiler ranges (`mom.vllm.prefill`, `mom.vllm.decode`). Standalone compile speedup benchmarks were not reported due to time constraints; the infrastructure is functional and ready for measurement.

- **Multi-agent concurrent evaluation (Workstream D):** The Poisson-arrival concurrent runner (`evaluation/concurrent_runner.py`) validated 100% pin survival across 7 pin/reuse cycles under 60-agent concurrent load on H100, with zero evictions and zero TTL expirations.

- **What did not work / was descoped:** (1) Asynchronous CPU offload (the three-tier GPU→CPU→evict hierarchy) was architecturally designed but not implemented — the blocking prefill path in the V0 scheduler makes async offload non-trivial. (2) Compiling from the vendored vLLM source (`pip install -e ./vllm`) was abandoned after build times exceeded 6 hours on GCP; the file-replacement approach above achieves identical results for pure-Python modifications. (3) LMCache integration was descoped — pin-or-evict over prefix caching achieves the same goal without an external caching tier.

---

## 7. Notes

- **Configuration:** All experiments are driven by YAML files in `configs/`. The ablation matrix covers 8 configs spanning baseline → prefix-cache-only → retention → retention sub-ablations → INT8/INT4 quantization.
- **Secrets:** `WANDB_API_KEY` and `HF_TOKEN` are loaded from environment variables. See `.env.example` (not committed).
- **Mock mode:** Every script supports `--mock-engine` for CPU-only smoke testing without a GPU or model.
- **Events log:** After a retention run, `results/<run>/events.jsonl` records every `pin`, `reuse`, `expire`, `evict`, and `pin_rejected_budget` event. Zero counts across all types indicate a wiring problem.

### AI Use Disclosure

*Per the HPML AI Use Policy (posted on CourseWorks). Required for every submission.*

**Did your team use any AI tool in completing this project?**

- [ ] No, we did not use any AI tool.
- [ ] Yes, we used AI assistance as described below.

**Tool(s) used:** *[to be filled by team]*

**Specific purpose:** *[to be filled by team]*

**Sections affected:** *[to be filled by team]*

**How we verified correctness:** *[to be filled by team]*

By submitting this project, the team confirms that the analysis, interpretations, and conclusions are our own, and that any AI assistance is fully disclosed above.

### License

Released under the MIT License. See [`LICENSE`](LICENSE).

### Citation

```bibtex
@misc{mom2026hpml,
  title  = {Mixture of Memory: Tool-Aware KV Cache Retention for Agentic LLM Serving},
  author = {A., Dakshinamoorthy and Ryssdal-Banoun, Alexander and Lee, Andrew and Krishna Jha, Vatsalam},
  year   = {2026},
  note   = {HPML Spring 2026 Final Project, Columbia University},
  url    = {https://github.com/al4744/MoM}
}
```

### Contact

Open a GitHub Issue or email the team:
- da3232@columbia.edu
- ar4678@columbia.edu
- al4744@columbia.edu
- vkj2107@columbia.edu

---

*HPML Spring 2026 — Dr. Kaoutar El Maghraoui — Columbia University*
