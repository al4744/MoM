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

PYTHONPATH=. "$PYTHON_BIN" benchmarks/trace_generator.py \
  --turns 5 10 \
  --context-lengths 1024 4096 \
  --output results_dry/workstream_c_traces

PYTHONPATH=. "$PYTHON_BIN" benchmarks/run_compile_benchmarks.py \
  --config configs/baseline.yaml \
  --output-root results_dry/workstream_c_compile_smoke \
  --context-lengths 1024 4096 \
  --turns 5 \
  --mock-engine
