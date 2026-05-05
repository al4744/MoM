#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pass() {
  echo "OK: $1"
}

todo() {
  echo "TODO: $1"
}

warn() {
  echo "WARNING: $1"
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ -e "$path" ]]; then
    pass "$label"
  else
    todo "$label missing"
  fi
}

check_path README.md "README exists"
if [[ -f requirements.txt || -f environment.yml ]]; then
  pass "requirements.txt or environment.yml exists"
else
  todo "requirements.txt or environment.yml missing"
fi
check_path configs "configs/ exists"
check_path scripts "scripts/ exists"
if [[ -d src ]]; then
  pass "src/ exists"
else
  todo "src/ missing or not applicable"
fi
check_path evaluation/tests "evaluation tests exist"
check_path deliverables "deliverables/ exists"

if compgen -G "deliverables/*.pdf" >/dev/null 2>&1; then
  pass "final report PDF found in deliverables/"
else
  todo "final report PDF not found in deliverables/"
fi

if compgen -G "deliverables/*.pptx" >/dev/null 2>&1 || \
   compgen -G "deliverables/*.key" >/dev/null 2>&1 || \
   compgen -G "deliverables/*presentation*.pdf" >/dev/null 2>&1 || \
   compgen -G "deliverables/*slides*.pdf" >/dev/null 2>&1; then
  pass "presentation file found in deliverables/"
else
  todo "presentation file not found in deliverables/"
fi

if grep -q "Experiment Tracking" README.md; then
  pass "README contains Experiment Tracking section"
else
  todo "README missing Experiment Tracking section"
fi

if grep -Eq "TODO:.*WandB|wandb\.ai|WandB dashboard URL" README.md; then
  pass "README contains a TODO or real WandB URL"
else
  todo "README missing WandB URL TODO or real URL"
fi

secret_pattern='(WANDB_API_KEY[[:space:]]*=[[:space:]]*[A-Za-z0-9_-]{20,}|api[_-]?key[[:space:]]*[:=][[:space:]]*["'\'']?[A-Za-z0-9_-]{20,}|secret[_-]?key[[:space:]]*[:=][[:space:]]*["'\'']?[A-Za-z0-9_-]{20,}|-----BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----)'
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if git grep -InE "$secret_pattern" -- . ':(exclude)scripts/check_submission_ready.sh' >/dev/null 2>&1; then
    warn "possible tracked secrets found; inspect git grep output"
    git grep -InE "$secret_pattern" -- . ':(exclude)scripts/check_submission_ready.sh' || true
  else
    pass "no obvious secrets found in tracked files"
  fi

  large_files=$(git ls-files -s | awk '$4 != "" {print $4}' | while read -r path; do
    if [[ -f "$path" ]]; then
      size=$(wc -c < "$path")
      if [[ "$size" -gt 104857600 ]]; then
        echo "$path $size"
      fi
    fi
  done)
  if [[ -n "$large_files" ]]; then
    warn "tracked files over 100 MB:"
    echo "$large_files"
  else
    pass "no tracked files over 100 MB"
  fi

  staged_large=$(git diff --cached --name-only | while read -r path; do
    if [[ -f "$path" ]]; then
      size=$(wc -c < "$path")
      if [[ "$size" -gt 104857600 ]]; then
        echo "$path $size"
      fi
    fi
  done)
  if [[ -n "$staged_large" ]]; then
    warn "staged files over 100 MB:"
    echo "$staged_large"
  else
    pass "no staged files over 100 MB"
  fi

  echo "Git status summary:"
  git status --short
else
  warn "not inside a Git work tree; tracked/staged checks unavailable"
  if rg -n "$secret_pattern" --glob '!scripts/check_submission_ready.sh' . >/dev/null 2>&1; then
    warn "possible secrets found in working files; inspect rg output"
    rg -n "$secret_pattern" --glob '!scripts/check_submission_ready.sh' . || true
  else
    pass "no obvious secrets found in working files"
  fi
  find . -type f -size +100M -print | while read -r path; do
    warn "file over 100 MB in working tree: $path"
  done
  echo "Git status summary: unavailable"
fi
