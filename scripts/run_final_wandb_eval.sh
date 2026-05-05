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

"$PYTHON_BIN" - <<'PY'
try:
    import vllm  # noqa: F401
except Exception as exc:
    raise SystemExit(f"vLLM is not importable. Run bash scripts/setup_vm_env.sh first. Detail: {exc}")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "No GPU detected: nvidia-smi is not available. Final real runs require the GPU/vLLM VM." >&2
  exit 1
fi

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "No GPU detected by nvidia-smi. Final real runs require the GPU/vLLM VM." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
try:
    import wandb  # noqa: F401
except Exception as exc:
    raise SystemExit(f"WandB is not importable. Run bash scripts/setup_vm_env.sh first. Detail: {exc}")
PY

if [[ -z "${WANDB_API_KEY:-}" ]] && ! { [[ -f "${HOME}/.netrc" ]] && grep -q "api.wandb.ai" "${HOME}/.netrc"; }; then
  echo "Please run: wandb login --relogin"
  exit 1
fi

mkdir -p results/wandb/baseline results/wandb/retention

run_real_eval() {
  local config="$1"
  local output="$2"
  local tags="$3"

  if [[ -n "${WANDB_ENTITY:-}" ]]; then
    PYTHONPATH=. "$PYTHON_BIN" evaluation/run_eval.py \
      --config "$config" \
      --output "$output" \
      --wandb \
      --wandb-mode online \
      --wandb-project mom-eval \
      --wandb-group final-real \
      --wandb-tags "$tags" \
      --wandb-entity "$WANDB_ENTITY"
  else
    PYTHONPATH=. "$PYTHON_BIN" evaluation/run_eval.py \
      --config "$config" \
      --output "$output" \
      --wandb \
      --wandb-mode online \
      --wandb-project mom-eval \
      --wandb-group final-real \
      --wandb-tags "$tags"
  fi
}

run_real_eval configs/baseline.yaml results/wandb/baseline baseline,real
run_real_eval configs/retention.yaml results/wandb/retention retention,optimized,real
