#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util
import subprocess
import sys

missing = []
for package, module in (("PyYAML", "yaml"), ("wandb", "wandb")):
    if importlib.util.find_spec(module) is None:
        missing.append(package)

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

if [[ -d ./vllm ]]; then
  python -m pip install -e ./vllm
fi

python - <<'PY'
import platform
import sys

version = sys.version_info
print(f"Python: {platform.python_version()}")
if (version.major, version.minor) not in {(3, 11), (3, 12)}:
    print("WARNING: Python 3.11 or 3.12 is recommended for the GPU/vLLM VM.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found; CUDA/GPU diagnostics are unavailable."
fi
