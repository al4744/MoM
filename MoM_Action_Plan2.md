# Mixture of Memory (MoM) — 4-Week Action Plan

**Project:** Specialized Memory Experts for Agentic LLM Serving
**Course:** COMS E6998, High Performance Machine Learning, Spring 2026
**Team:** Dakshinamoorthy A, Alexander Ryssdal-Banoun, Andrew Lee, Vatsalam Krishna Jha
**Timeline:** March 27 – April 27, 2026

---

## Infrastructure

The team works on a single shared GCP VM (2–4x A100 40GB, 128–256GB CPU RAM) with everyone connecting via SSH. Provision at least 200–300 GB of SSD — between the OS, Python environment, vLLM, LMCache, both models (~30 GB total for Llama 3 8B + Mistral 7B in FP16), and PyTorch Profiler traces, a default 100 GB boot disk will fill up fast. Adding persistent SSD is cheap (~$0.17/GB/month). During Week 2 peak parallelism, consider a second smaller spot instance as overflow for GPU contention. Use a lightweight Slack-based claim system for GPU time.

---

## Workstreams

The project splits into five tracks. Three are the core optimization techniques (A–C) that can be developed in parallel. A fourth (D) owns the end-to-end evaluation across all of them. The fifth (E) is the MoM extension — a research stretch goal that operates at the prompt-assembly level, decoupled from vLLM internals.

### Workstream A — vLLM KV Retention Policy

The systems-heavy core of the project. Builds on top of vLLM's existing KV Offloading Connector (not deep scheduler modifications) to minimize engineering risk while testing the core hypothesis. Involves building the latency predictor (EMA-based heuristic that classifies tool calls into pin/offload/evict tiers) and implementing the async offload/reload pipeline. Must preserve KV validity and correctness at all times — vLLM's preemption and swapping logic is tightly coupled, so changes require careful validation. Also includes the LMCache comparison, since this person will understand caching tier tradeoffs most deeply.

### Workstream B — KV Cache Quantization

INT8/INT4 quantization of KV cache states (not model weights). Selective retention: full precision for recent turns, INT8 for older referenced turns, drop superseded context. Accuracy-vs-memory tradeoff measurement at each quantization level. Self-contained enough to develop and benchmark independently before integrating with the retention pipeline. Naturally owns the AgentBench/ToolBench accuracy evaluations.

### Workstream C — torch.compile + Profiling Infrastructure

torch.compile applied to the inference pipeline (prefill and decode independently), benchmarked across context lengths. Paired with owning the profiling and benchmarking infrastructure: synthetic trace design, automated benchmark harness, WandB setup, PyTorch Profiler and NVIDIA Nsight instrumentation. These pair naturally because measuring torch.compile gains requires deep profiling tooling anyway.

### Workstream D — Evaluation Lead

End-to-end experimental narrative: metrics suite definition, WandB dashboard, full evaluation matrix across all configurations and turn counts, final comparison tables. This role has natural visibility into how all the pieces fit together and is critical for producing a credible paper.

### Workstream E — MoM Extension (Stretch Goal)

Training the routing MLP (~1M params, ~5K labeled turns), implementing the four memory experts (Factual/BM25, Episodic/causal-chain, Semantic/FAISS, Procedural/trajectory-matching), and the prompt-assembly pipeline. **Important guardrail:** MoM operates strictly at the prompt-assembly level — it selects and injects context into the prompt before it reaches vLLM. It does not touch vLLM's paged block table or attempt to splice KV entries directly (instructor feedback confirms this would be much harder than it sounds and risks breaking layer/position alignment). MoM won't block Workstreams A–C. This is the project's research extension — high upside but can be scoped down or cut if the core work needs more attention.

---

## Week 1: March 27 – April 2 — Environment Setup & Baseline Profiling

**Goal:** Full baseline stack running, initial profiling data collected, problem confirmed measurable.

### All Team

- [ ] Set up shared GitHub repo with branch strategy (main → dev → feature branches)
- [ ] Provision GCP instances (2–4x A100 40GB), confirm SSH access for everyone
- [ ] Install core stack: vLLM (latest), PyTorch 2.x, LangChain, WandB, LMCache
- [ ] Download models: Llama 3 8B (primary), Mistral 7B (comparison)
- [ ] Agree on communication cadence: daily async Slack updates, 2x/week syncs

### Workstream A

- [ ] Deep-dive into vLLM's KV Offloading Connector source code — map the load/store/eviction paths and the CPU backend for native offloading
- [ ] Understand vLLM's preemption, swapping, and scheduling pipeline as it relates to KV validity — document where preemptions happen and what triggers recomputation
- [ ] Confirm that the Connector API exposes sufficient control for per-request pin/offload/evict decisions without forking the scheduler (if not, document the gap and propose minimal patches)
- [ ] Set up a dev branch for KV retention modifications

### Workstream B

- [ ] Research INT8/INT4 quantization approaches for KV cache states (not model weights)
- [ ] Survey existing KV cache quantization implementations (AWQ, GPTQ patterns adapted to KV)
- [ ] Begin accuracy baseline: run AgentBench subset to get task-accuracy reference numbers

### Workstream C

- [ ] Design synthetic trace generator: parameterized by prompt length, tool call frequency, latency distribution, and turn count (5/10/25/50 turns)
- [ ] Define the full metrics suite: prefill recomputation time after tool return, vLLM preemption count, TTFT, TBT, peak VRAM, KV hit/eviction rates, CPU↔GPU transfer bandwidth
- [ ] Set up WandB project with experiment tracking structure
- [ ] Set up torch.compile with Llama 3 8B and verify it runs on A100 — capture initial compile-time vs. runtime tradeoffs
- [ ] Run **baseline profiling** on vanilla vLLM with synthetic traces at all 4 turn counts
- [ ] Capture PyTorch Profiler traces and NVIDIA Nsight GPU timelines for baseline

### Workstream D

- [ ] Define WandB experiment naming conventions and dashboard structure

### Workstream E (MoM Extension)

- [ ] Read and annotate key MoM papers: MoE (Shazeer 2017), MemGPT, Scissorhands, H₂O
- [ ] Design the MoM routing architecture: input features, expert definitions, training data plan
- [ ] Begin collecting/labeling GPT-4 data for the ~5K labeled turns (Factual / Episodic / Semantic / Procedural)
- [ ] Explore FAISS setup for the Semantic expert's vector similarity store

### Week 1 Milestone
> **Deliverable:** Baseline profiling report showing recomputation overhead at 5/10/25/50 turns. All team members can independently run experiments on GCP. Shared repo and WandB are operational.

---

## Week 2: April 3 – April 9 — Core Implementation

**Goal:** Implement the three main optimization techniques in parallel; MoM extension underway.

### Workstream A — KV Retention

- [ ] Implement the **latency predictor** (heuristic lookup, EMA-updated) that classifies tool calls into three tiers:
  - Fast tools (<500ms): pin KV blocks in VRAM
  - Medium tools (500ms–2s): offload to CPU asynchronously
  - Long tools (>2s): allow preemption under VRAM pressure
- [ ] Build the asynchronous offload/reload pipeline on top of vLLM's KV Offloading Connector
- [ ] Implement pinning logic so that pinned blocks resume instantly on tool return
- [ ] Ensure offloaded blocks reload overlapped with I/O (no blocking on the critical path)
- [ ] Write unit tests verifying KV validity is preserved across offload/reload cycles

### Workstream B — Quantization

- [ ] Implement INT8 quantization of cached KV states with selective retention:
  - Full precision for recent turns
  - INT8 for older referenced turns
  - Drop superseded context
- [ ] Implement INT4 as an aggressive alternative, measure accuracy tradeoff
- [ ] Measure accuracy vs. memory tradeoff per quantization level on AgentBench subset

### Workstream C — torch.compile + Profiling

- [ ] Apply torch.compile to the inference pipeline (prefill and decode separately)
- [ ] Benchmark torch.compile gains independently across context lengths (1K, 4K, 8K, 16K tokens)
- [ ] Build automated benchmarking scripts that run the full trace suite and log to WandB
- [ ] Create the comparison dashboard in WandB: vanilla vs. optimized, with panels for TTFT, prefill recomp time, VRAM utilization, CPU↔GPU bandwidth
- [ ] Profile Workstream A's KV retention implementation as it becomes testable
- [ ] Begin integrating ToolBench traces as a secondary, more realistic workload
- [ ] Set up PyTorch Profiler overlay scripts for GPU time breakdown (prefill / decode / cache management)

### Workstream D — Evaluation

- [ ] Begin profiling Workstreams A and B outputs as they become available

### Workstream E — MoM Extension

- [ ] Train the routing MLP (~1M params) on the labeled turn dataset
- [ ] Implement the four memory experts:
  - **Factual:** BM25 index over structured KV store
  - **Episodic:** Causal chain + recency scoring over temporal log
  - **Semantic:** INT8 FAISS vector similarity lookup
  - **Procedural:** Trajectory matching on action sequences
- [ ] Build the prompt-assembly pipeline: router selects expert → expert retrieves context → context injected into prompt
- [ ] Unit test each expert independently with synthetic queries

### Week 2 Milestone
> **Deliverable:** All three core optimizations (KV retention, KV quantization, torch.compile) functional individually. MoM router trained and experts implemented. Ready for integration testing.

---

## Week 3: April 10 – April 16 — Integration, Evaluation & LMCache Comparison

**Goal:** Combine optimizations, run the full evaluation suite, benchmark against LMCache, evaluate MoM.

### Workstreams A + B (Integration)

- [ ] Combine KV retention + KV quantization into a single pipeline (quantize offloaded blocks, keep pinned blocks in full precision)
- [ ] Layer torch.compile on top of the combined system
- [ ] Stress test: 50-turn conversations with mixed tool latencies — verify no KV corruption
- [ ] Profile the integrated system end-to-end: confirm targets (1.5–2.5x TTFT reduction, <2% accuracy drop)

### Workstream A — LMCache Comparison

- [ ] Set up LMCache with GPU/CPU/disk/remote tiers
- [ ] Run head-to-head benchmarks: LMCache vs. our retention policy on identical traces
- [ ] Evaluate whether dedicated tiering infrastructure adds gains beyond vLLM's native connector

### Workstream D — Full Evaluation

- [ ] Run the complete benchmark suite on all configurations:
  - Vanilla vLLM (baseline)
  - KV retention only
  - KV retention + INT8 quantization
  - KV retention + INT4 quantization
  - Full stack (retention + quantization + torch.compile)
  - LMCache (GPU/CPU/disk/remote tiers)
- [ ] Collect metrics across all turn counts (5/10/25/50) for each configuration
- [ ] Run AgentBench and ToolBench evaluations for task accuracy
- [ ] Generate comparison tables and WandB visualizations
- [ ] Run Mistral 7B experiments for cross-model generalization

### Workstream E — MoM Evaluation

- [ ] Integrate MoM routing with the optimized serving stack
- [ ] Evaluate MoM's context reduction: effective context size vs. full conversation history
- [ ] Benchmark MoM routing accuracy and overhead (routing latency, expert retrieval time)
- [ ] Compare: optimized stack alone vs. optimized stack + MoM

### Week 3 Milestone
> **Deliverable:** Full experimental results across all configurations. LMCache comparison complete. MoM extension evaluated. All numbers needed for the final report are collected.

---

## Week 4: April 17 – April 27 — Paper, Demo & Polish

**Goal:** Write the final report, build the demo, and prepare for submission.

### April 17–20: Writing Sprint

| Section | Owner | Details |
|---------|-------|---------|
| Abstract + Introduction | Workstream D | Problem statement, motivation, contributions |
| Background & Related Work | TBD | vLLM, PagedAttention, KV caching, MoE, MemGPT |
| System Design (KV Retention) | Workstream A | Retention policy, latency predictor, offload pipeline |
| KV Quantization + torch.compile | Workstreams B + C | Quantization approach, selective retention, compile gains |
| MoM Extension | Workstream E | Routing architecture, expert designs, prompt assembly |
| Evaluation | Workstream D | Experimental setup, results, analysis, comparisons |
| Conclusion | Workstream A | Summary, limitations, future work |

### April 21–23: Demo Development

- [ ] Build the live WandB dashboard — side-by-side vanilla vs. optimized stack showing TTFT per tool call, prefill recomputation time, VRAM utilization
- [ ] Set up the live agent demo (vanilla vLLM vs. optimized) with a representative multi-turn tool-use conversation
- [ ] Create the LMCache comparison panel in the dashboard
- [ ] Prepare MoM routing visualization (which expert handles which query type)

### April 24–25: Review & Revision

- [ ] Full team paper review — each person reads and comments on all sections
- [ ] Cross-check all numbers in the paper against WandB logs
- [ ] Proofread, fix figures, ensure consistent notation
- [ ] Dry-run the demo end-to-end

### April 26–27: Final Submission

- [ ] Final paper PDF compiled and submitted
- [ ] Code repo cleaned up with README, requirements.txt, and reproduction instructions
- [ ] Demo materials finalized
- [ ] All WandB experiments tagged and organized for reference

---

## Design Considerations (from Instructor Feedback)

The following points reflect instructor feedback on the proposal and should guide implementation decisions throughout the project.

### Scope management: one project, not two

The core deliverable is tool-aware KV handling for paused agent requests (pin vs. offload vs. recompute). MoM (the routing MLP + memory experts) is an optional extension. Workstreams A–D are the core project; Workstream E is MoM. If the core work needs more time, MoM gets scoped down or cut. The paper stands strong on KV retention + quantization + torch.compile alone.

### Start with the KV Offloading Connector, not the scheduler

vLLM's scheduler/block manager logic around preemption, swapping, and KV validity is intricate. Modifying it directly carries high risk of breaking correctness or throughput. Build the first version of tool-aware retention entirely on top of vLLM's KV Offloading Connector, which already supports asynchronous KV load/store with a CPU backend. This is enough to test the core hypothesis — that tool-call pauses should not force expensive reprefill — with far less engineering risk. Only escalate to scheduler modifications if the Connector API hits a hard wall, and document exactly what was changed and why.

### MoM should not touch vLLM internals

"Injecting router-chosen KV entries into vLLM's paged block table" is much harder than it sounds — vLLM's KV blocks are internal model state, not an external memory that can be arbitrarily spliced. MoM should remain scoped to prompt-level assembly (selecting and injecting context into the prompt before it reaches vLLM), not direct KV block manipulation.

### Synthetic traces as primary benchmark

AgentBench may conflate "agent capability" with "systems effects." The primary performance benchmark should be synthetic, controlled traces with fixed prompt lengths, fixed tool-call frequency, and fixed tool latency distributions. This isolates the "pause → resume" pattern and lets us attribute gains cleanly to KV retention/offload rather than prompt differences. ToolBench and AgentBench serve as realism layers, not primary evidence.

### Metrics should go beyond TTFT/TBT

In addition to TTFT and TBT, the evaluation should report: (1) prefill recomputation time after tool return, (2) number of vLLM preemptions triggered by KV pressure, and (3) CPU↔GPU transfer time/bandwidth when offloading is enabled. These should be treated as first-class metrics in the paper, not supplementary.

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| GCP GPU quota delays | Apply for quota increase Day 1; have Columbia Terremoto as backup |
| KV retention breaks vLLM correctness | Unit tests for KV validity at every checkpoint; compare outputs against vanilla vLLM. vLLM's preemption/swapping is tightly coupled — validate after every change |
| Connector API insufficient for retention policy | Start Connector-only (Week 1 gap analysis). Only escalate to minimal scheduler patches if the Connector can't support per-request pin/offload decisions. Document every change |
| AgentBench conflates agent capability with systems effects | Use synthetic controlled traces as primary benchmark (fixed prompts, fixed tool frequency, fixed latency). AgentBench/ToolBench are realism layers, not primary evidence |
| MoM training data insufficient | Start labeling Week 1; use GPT-4 for semi-automated labeling; if quality is low, simplify to 2–3 experts instead of 4 |
| MoM scope creep into vLLM internals | Hard guardrail: MoM operates at prompt assembly only — no direct KV block manipulation. If routing requires KV-level control, descope to prompt-level context selection |
| torch.compile incompatibility with vLLM | Isolate torch.compile to inference-only path; can be evaluated independently even if it can't be combined |
| Integration issues in Week 3 | Keep each optimization modular and independently evaluable — the paper is still strong with individual results even if full integration has issues |

---

## Communication Plan

- **Daily:** Async Slack updates (what you did, what's next, any blockers)
- **Tuesday & Friday:** 30-min video sync to review progress and unblock issues
- **WandB:** All experiments logged with clear naming conventions (`baseline-llama3-50turn`, `kv-retention-int8-50turn`, etc.)
- **GitHub:** PRs required for merging to dev; code reviews by at least one other team member

---

## Deliverables Checklist

- [ ] Final report (conference-style paper, ~8 pages)
- [ ] GitHub repo with reproducible code
- [ ] WandB dashboard with all experimental results
- [ ] Live demo: vanilla vs. optimized vLLM agent serving
- [ ] Profiling artifacts (PyTorch Profiler traces, Nsight timelines)
