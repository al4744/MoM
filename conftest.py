"""Repo-root conftest.

Adds the repo root to sys.path so `from evaluation.metrics import ...` and
`from src.retention.config import ...` work without a pyproject.toml install.
This matches the existing import style used in src/retention/ tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
