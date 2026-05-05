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

PYTHON ?= python
RESULTS ?= results_dry
REAL_RESULTS ?= results_quant
CONFIGS := $(wildcard configs/*.yaml)

.PHONY: test test-quick eval-baseline eval-retention eval-all eval-all-real smoke smoke-baseline smoke-retention compare compare-real ablate ablate-real clean

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
# Housekeeping
# ----------------------------------------------------------------------------

clean:
	rm -rf $(RESULTS) **/__pycache__ .pytest_cache
