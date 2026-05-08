# =============================================================================
# Workstream D — Make targets for the evaluation pipeline.
# =============================================================================
# Usage:
#   make test          # run unit tests
#   make eval-baseline # dry-run baseline config
#   make eval-all      # dry-run every config under configs/  → results_dry/
#   make compare       # cross-config comparison table from results_dry/
#   make ablate A=baseline B=retention   # pairwise delta
#   make eval-all-real # real runs for quant configs          → results_quant/
#   make compare-real  # comparison table from results_quant/
#   make clean         # remove results_dry/, results_quant/ and __pycache__
# =============================================================================

PYTHON ?= python3
RESULTS ?= results_dry
REAL_RESULTS ?= results_quant
CONFIGS := $(wildcard configs/*.yaml)

.PHONY: test test-quick eval-baseline eval-retention eval-all eval-all-real smoke smoke-baseline smoke-retention workstream-c-smoke compare compare-real ablate ablate-real clean \
        eval-concurrent eval-filler-focal eval-filler-focal-large \
        eval-tier2 eval-tier2-lockstep eval-tier2-staggered eval-tier2-heterogeneous eval-tier2-burst eval-tier2-filler-focal eval-tier2-large \
        eval-accuracy-baseline eval-accuracy-retention eval-accuracy-int8 eval-accuracy-int4 eval-accuracy-all \
        eval-comprehensive eval-comprehensive-large

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

test:
	PYTHONPATH=. $(PYTHON) -m pytest evaluation/tests/ src/retention/tests/ src/quantization/tests/ -v

test-quick:
	PYTHONPATH=. $(PYTHON) -m pytest evaluation/tests/ src/retention/tests/ src/quantization/tests/ -q

# ----------------------------------------------------------------------------
# Eval runs (dry-run mode until A/B/C engine hooks land)
# ----------------------------------------------------------------------------

eval-baseline:
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/baseline.yaml \
		--output $(RESULTS)/baseline \
		--dry-run

eval-retention:
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/retention.yaml \
		--output $(RESULTS)/retention \
		--dry-run

# Smoke test: real run_eval pipeline, fake engine, no CUDA, no vLLM.
# Useful for proving the metrics pipeline before hitting GCP.
smoke-baseline:
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/baseline.yaml \
		--output $(RESULTS)/baseline-smoke \
		--mock-engine

smoke-retention:
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/retention.yaml \
		--output $(RESULTS)/retention-smoke \
		--mock-engine

smoke: smoke-baseline smoke-retention
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--ablate $(RESULTS)/baseline-smoke/summary.json $(RESULTS)/retention-smoke/summary.json \
		--output $(RESULTS)/smoke-ablate.md

workstream-c-smoke:
	bash scripts/run_workstream_c_smoke.sh

# Iterate every YAML config under configs/ and dry-run it.
eval-all:
	@for cfg in $(CONFIGS); do \
		name=$$(basename $$cfg .yaml); \
		echo "==> $$name"; \
		PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
			--config $$cfg \
			--output $(RESULTS)/$$name \
			--dry-run; \
	done

# Non-dry evaluation matrix for the main paper configs.
eval-all-real:
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/baseline.yaml \
		--output $(REAL_RESULTS)/baseline
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/retention.yaml \
		--output $(REAL_RESULTS)/retention
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/retention_int8.yaml \
		--output $(REAL_RESULTS)/retention_int8
	PYTHONPATH=. $(PYTHON) evaluation/run_eval.py \
		--config configs/retention_int4.yaml \
		--output $(REAL_RESULTS)/retention_int4

# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

compare:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--results-root $(RESULTS) \
		--output $(RESULTS)/comparison.md
	@echo ""
	@cat $(RESULTS)/comparison.md

compare-real:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--results-root $(REAL_RESULTS) \
		--output $(REAL_RESULTS)/comparison.md
	@echo ""
	@cat $(REAL_RESULTS)/comparison.md

# Pairwise delta. Usage: make ablate A=baseline B=retention
ablate:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--ablate $(RESULTS)/$(A)/summary.json $(RESULTS)/$(B)/summary.json \
		--output $(RESULTS)/ablate-$(A)-vs-$(B).md
	@echo ""
	@cat $(RESULTS)/ablate-$(A)-vs-$(B).md

# Pairwise delta for real (non-dry) runs.
ablate-real:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--ablate $(REAL_RESULTS)/$(A)/summary.json $(REAL_RESULTS)/$(B)/summary.json \
		--output $(REAL_RESULTS)/ablate-$(A)-vs-$(B).md
	@echo ""
	@cat $(REAL_RESULTS)/ablate-$(A)-vs-$(B).md

# ----------------------------------------------------------------------------
# Concurrent multi-agent benchmark
# ----------------------------------------------------------------------------
# Usage:
#   make eval-concurrent CONFIG=configs/retention_constrained.yaml CONCURRENCY=4
#
# Drives N concurrent traces through one engine. Each YAML trace shape is
# replicated `--num-traces` times (defaults to CONCURRENCY). Output shape is
# identical to run_eval; comparison_table.py works unchanged.

CONFIG ?= configs/retention_constrained.yaml
CONCURRENCY ?= 2
CONCURRENT_RESULTS ?= results/concurrent

eval-concurrent:
	@cfgname=$$(basename $(CONFIG) .yaml); \
	out=$(CONCURRENT_RESULTS)/$$cfgname-c$(CONCURRENCY); \
	echo "=== Concurrent run: $$cfgname @ concurrency=$(CONCURRENCY) ==="; \
	echo "  output: $$out"; \
	PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
		--config $(CONFIG) \
		--output $$out \
		--concurrency $(CONCURRENCY)

# ----------------------------------------------------------------------------
# Tier 1 — filler+focal microbenchmark (Daksh's regime)
# ----------------------------------------------------------------------------
# Usage:
#   TS=$(date +%s) make eval-filler-focal TS=$$TS
#
# Runs the (baseline_filler, prefix_cache_only_filler, retention_filler) trio
# through the filler+focal workload pattern. The headline number is the
# (prefix_cache_only_filler → retention_filler) delta on focal post-tool TTFT.

NUM_FILLERS ?= 7
FOCAL_NUM_TURNS ?= 4
FOCAL_TOOL_LATENCY_MS ?= 5000
FILLER_RESULTS ?= results/filler-$(shell date +%Y%m%d-%H%M%S)

eval-filler-focal:
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		echo "==> $$cfg"; \
		PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
			--config configs/$$cfg.yaml \
			--output $(FILLER_RESULTS)/$$cfg \
			--workload-class filler_focal \
			--num-fillers $(NUM_FILLERS) \
			--focal-num-turns $(FOCAL_NUM_TURNS) \
			--focal-tool-latency-ms $(FOCAL_TOOL_LATENCY_MS) \
			--concurrency $$(($(NUM_FILLERS) + 1)); \
	done
	@echo ""
	@echo "=== Filler+focal results in $(FILLER_RESULTS)/ ==="
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		echo "  --- $$cfg ---"; \
		$(PYTHON) -c "import json, sys; \
			b=json.load(open('$(FILLER_RESULTS)/$$cfg/summary.json')); \
			w=b['workload']; \
			print('    focal post-tool ttft (ms):  mean=' + format(w['focal_post_tool_ttft']['mean_ms'], '.1f'), \
			      'p95=' + format(w['focal_post_tool_ttft']['p95_ms'], '.1f'), \
			      'p99=' + format(w['focal_post_tool_ttft']['p99_ms'], '.1f'))"; \
	done

# ----------------------------------------------------------------------------
# Tier 2 — full 5-class workload battery
# ----------------------------------------------------------------------------
# Runs each of the 5 workload classes against (baseline_filler,
# prefix_cache_only_filler, retention_filler) as engine configs. Each class
# tests a different cross-agent contention pattern.
#
# Usage:
#   make eval-tier2

NUM_AGENTS ?= 8
NUM_USER_PROMPTS ?= 4
TOOL_LATENCY_MS ?= 2000
# Arrival rate for staggered (Poisson). Default 0.5/s is calibrated for the
# 8-agent regime; bump to 5.0/s when running at NUM_AGENTS≥30 so agents
# actually overlap (otherwise total spread ≫ per-agent runtime → effectively
# sequential, no contention).
ARRIVAL_RATE_PER_SEC ?= 0.5
# Burst window for burst (uniform). Default 500ms keeps an 8-agent burst
# tight; bump to 5000ms at scale so some agents are mid-prefill while others
# enter tool gaps (otherwise everyone pauses together → no eviction force).
BURST_DURATION_MS ?= 500
HETERO_LOG_MEAN_MS ?= 1500
HETERO_LOG_SIGMA ?= 0.7
TIER2_RESULTS ?= results/tier2-$(shell date +%Y%m%d-%H%M%S)

eval-tier2-lockstep:
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
			--config configs/$$cfg.yaml \
			--output $(TIER2_RESULTS)/lockstep/$$cfg \
			--workload-class lockstep \
			--num-agents $(NUM_AGENTS) \
			--num-user-prompts $(NUM_USER_PROMPTS) \
			--tool-latency-ms $(TOOL_LATENCY_MS) \
			--concurrency $(NUM_AGENTS); \
	done

eval-tier2-staggered:
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
			--config configs/$$cfg.yaml \
			--output $(TIER2_RESULTS)/staggered/$$cfg \
			--workload-class staggered \
			--num-agents $(NUM_AGENTS) \
			--num-user-prompts $(NUM_USER_PROMPTS) \
			--tool-latency-ms $(TOOL_LATENCY_MS) \
			--arrival-rate-per-sec $(ARRIVAL_RATE_PER_SEC) \
			--concurrency $(NUM_AGENTS); \
	done

eval-tier2-heterogeneous:
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
			--config configs/$$cfg.yaml \
			--output $(TIER2_RESULTS)/heterogeneous/$$cfg \
			--workload-class heterogeneous \
			--num-agents $(NUM_AGENTS) \
			--num-user-prompts $(NUM_USER_PROMPTS) \
			--tool-latency-log-mean-ms $(HETERO_LOG_MEAN_MS) \
			--tool-latency-log-sigma $(HETERO_LOG_SIGMA) \
			--concurrency $(NUM_AGENTS); \
	done

eval-tier2-burst:
	@for cfg in baseline_filler prefix_cache_only_filler retention_filler; do \
		PYTHONPATH=. $(PYTHON) scripts/run_concurrent_eval.py \
			--config configs/$$cfg.yaml \
			--output $(TIER2_RESULTS)/burst/$$cfg \
			--workload-class burst \
			--num-agents $(NUM_AGENTS) \
			--num-user-prompts $(NUM_USER_PROMPTS) \
			--tool-latency-ms $(TOOL_LATENCY_MS) \
			--burst-duration-ms $(BURST_DURATION_MS) \
			--concurrency $(NUM_AGENTS); \
	done

eval-tier2-filler-focal:
	@$(MAKE) eval-filler-focal FILLER_RESULTS=$(TIER2_RESULTS)/filler_focal

eval-tier2: eval-tier2-lockstep eval-tier2-staggered eval-tier2-heterogeneous eval-tier2-burst eval-tier2-filler-focal
	@echo ""
	@echo "=== Tier 2 battery complete ==="
	@echo "  results: $(TIER2_RESULTS)"

# ----------------------------------------------------------------------------
# H100-scale convenience targets
# ----------------------------------------------------------------------------
# Pre-calibrated for ~80GB GPUs running 50–60 concurrent agents at
# gpu_memory_utilization=0.85. Reproduces the conditions retention was
# designed for (sustained eviction pressure, real concurrent multi-agent
# contention) without manually overriding 6 different make variables.
#
# Usage:
#   make eval-filler-focal-large    # 60 fillers, 8 focal turns (n=7 post-tool)
#   make eval-tier2-large           # 50 agents, scale-appropriate spread
#   make eval-comprehensive-large   # filler+focal-large + tier2-large + accuracy

eval-filler-focal-large:
	@$(MAKE) eval-filler-focal \
		NUM_FILLERS=60 \
		FOCAL_NUM_TURNS=8 \
		FOCAL_TOOL_LATENCY_MS=5000

eval-tier2-large:
	@$(MAKE) eval-tier2 \
		NUM_AGENTS=50 \
		ARRIVAL_RATE_PER_SEC=5.0 \
		BURST_DURATION_MS=5000

eval-comprehensive-large: eval-filler-focal-large eval-tier2-large eval-accuracy-all
	@echo ""
	@echo "=== Comprehensive battery (H100-scale) complete ==="

# ----------------------------------------------------------------------------
# Accuracy battery — proxy for AgentBench / ToolBench reasoning capability.
# ----------------------------------------------------------------------------
# Runs lm-eval-harness on each retention/quant config, then diffs accuracy
# against baseline. ⚠ flag fires on any config losing >2 pp mean accuracy.

ACC_RESULTS ?= results/accuracy-$(shell date +%Y%m%d-%H%M%S)
ACC_TASK_SUITE ?= reasoning
ACC_LIMIT ?= 50

eval-accuracy-baseline:
	PYTHONPATH=. $(PYTHON) scripts/run_accuracy_eval.py \
		--config configs/baseline.yaml \
		--output $(ACC_RESULTS)/baseline \
		--task-suite $(ACC_TASK_SUITE) --limit $(ACC_LIMIT)

eval-accuracy-retention:
	PYTHONPATH=. $(PYTHON) scripts/run_accuracy_eval.py \
		--config configs/retention.yaml \
		--output $(ACC_RESULTS)/retention \
		--task-suite $(ACC_TASK_SUITE) --limit $(ACC_LIMIT)

eval-accuracy-int8:
	PYTHONPATH=. $(PYTHON) scripts/run_accuracy_eval.py \
		--config configs/retention_int8.yaml \
		--output $(ACC_RESULTS)/retention_int8 \
		--task-suite $(ACC_TASK_SUITE) --limit $(ACC_LIMIT)

eval-accuracy-int4:
	PYTHONPATH=. $(PYTHON) scripts/run_accuracy_eval.py \
		--config configs/retention_int4.yaml \
		--output $(ACC_RESULTS)/retention_int4 \
		--task-suite $(ACC_TASK_SUITE) --limit $(ACC_LIMIT)

eval-accuracy-all: eval-accuracy-baseline eval-accuracy-retention eval-accuracy-int8 eval-accuracy-int4
	PYTHONPATH=. $(PYTHON) scripts/compare_accuracy.py \
		$(ACC_RESULTS)/baseline \
		$(ACC_RESULTS)/retention \
		$(ACC_RESULTS)/retention_int8 \
		$(ACC_RESULTS)/retention_int4 \
		--baseline-name baseline | tee $(ACC_RESULTS)/comparison.md

# ----------------------------------------------------------------------------
# Comprehensive battery — Tier 1 + Tier 2 + Accuracy in one shot.
# ----------------------------------------------------------------------------
eval-comprehensive: eval-tier2 eval-accuracy-all
	@echo ""
	@echo "=== Comprehensive battery complete ==="
	@echo "  Tier 2 (workload classes): $(TIER2_RESULTS)"
	@echo "  Accuracy:                  $(ACC_RESULTS)"

# ----------------------------------------------------------------------------
# Housekeeping
# ----------------------------------------------------------------------------

clean:
	rm -rf $(RESULTS) **/__pycache__ .pytest_cache
