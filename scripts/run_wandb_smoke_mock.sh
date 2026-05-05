#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON+x}" && -x .venv/bin/python ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
else
  PYTHON_BIN="python3"
fi

if [[ "$PYTHON_BIN" == ".venv/bin/python" ]]; then
  source .venv/bin/activate
fi

echo "Mock WandB smoke runs validate logging only and are not final benchmark evidence."

mkdir -p results/wandb/baseline_mock results/wandb/retention_mock

run_mock_eval() {
  local config="$1"
  local output="$2"
  local tags="$3"

  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" evaluation/run_eval.py \
      --config "$config" \
      --output "$output" \
      --mock-engine \
      --wandb \
      --wandb-mode offline \
      --wandb-project mom-eval \
      --wandb-group smoke-mock \
      --wandb-tags "$tags" \
      --wandb-entity "$WANDB_ENTITY"
  else
    PYTHONPATH=. "$PYTHON_BIN" evaluation/run_eval.py \
      --config "$config" \
      --output "$output" \
      --mock-engine \
      --wandb \
      --wandb-mode offline \
      --wandb-project mom-eval \
      --wandb-group smoke-mock \
      --wandb-tags "$tags"
  fi
}

run_mock_eval configs/baseline.yaml results/wandb/baseline_mock baseline,mock,smoke
run_mock_eval configs/retention.yaml results/wandb/retention_mock retention,optimized,mock,smoke

echo "Mock runs complete. Do not use these outputs as final benchmark evidence."
