# =============================================================================
# Workstream D — Make targets for the evaluation pipeline.
# =============================================================================
# Usage:
#   make test          # run unit tests
#   make eval-baseline # dry-run baseline config
#   make eval-all      # dry-run every config under configs/
#   make compare       # cross-config comparison table from results/
#   make ablate A=baseline B=retention   # pairwise delta
#   make clean         # remove results/ and __pycache__
# =============================================================================

PYTHON ?= python
RESULTS ?= results
CONFIGS := $(wildcard configs/*.yaml)

.PHONY: test test-quick eval-baseline eval-retention eval-all compare ablate clean

# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

test:
	PYTHONPATH=. $(PYTHON) -m pytest evaluation/tests/ -v

test-quick:
	PYTHONPATH=. $(PYTHON) -m pytest evaluation/tests/ -q

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

# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

compare:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--results-root $(RESULTS) \
		--output $(RESULTS)/comparison.md
	@echo ""
	@cat $(RESULTS)/comparison.md

# Pairwise delta. Usage: make ablate A=baseline B=retention
ablate:
	PYTHONPATH=. $(PYTHON) evaluation/comparison_table.py \
		--ablate $(RESULTS)/$(A)/summary.json $(RESULTS)/$(B)/summary.json \
		--output $(RESULTS)/ablate-$(A)-vs-$(B).md
	@echo ""
	@cat $(RESULTS)/ablate-$(A)-vs-$(B).md

# ----------------------------------------------------------------------------
# Housekeeping
# ----------------------------------------------------------------------------

clean:
	rm -rf $(RESULTS) **/__pycache__ .pytest_cache
