"""Workstream D — Sanity check that every ablation config parses correctly.

Loads each config under configs/, builds the matching RetentionConfig
(or asserts None for non-retention configs), and prints a table summarising
the toggled flags. Exits non-zero on any parse failure.

Run before pushing new ablation configs:
    PYTHONPATH=. python scripts/check_ablation_configs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from evaluation.engine_adapter import build_retention_config


# Expected toggles per config. Used to verify YAML edits didn't drift.
EXPECTED: dict[str, dict] = {
    "baseline":             {"retention": False, "prefix_caching": False, "kv_quant": None},
    "prefix_cache_only":    {"retention": False, "prefix_caching": True,  "kv_quant": None},
    "retention":            {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                             "use_ema": True, "use_per_tool_ema": True, "pin_min": 0.10},
    "retention_no_ema":     {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                             "use_ema": False, "use_per_tool_ema": True, "pin_min": 0.10},
    "retention_no_per_tool":{"retention": True,  "prefix_caching": True,  "kv_quant": None,
                             "use_ema": True, "use_per_tool_ema": False, "pin_min": 0.10},
    "retention_no_pin":     {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                             "use_ema": True, "use_per_tool_ema": True, "pin_max": 0.01},
    "retention_int8":       {"retention": True,  "prefix_caching": True,  "kv_quant": "int8"},
    "retention_int4":       {"retention": True,  "prefix_caching": True,  "kv_quant": "int4"},
    "compile_profile":      {"retention": False, "prefix_caching": False, "kv_quant": None},
    # Memory-pressure variants (gpu_memory_utilization=0.77, tool_latency_ms=2000)
    "baseline_constrained":              {"retention": False, "prefix_caching": False, "kv_quant": None},
    "prefix_cache_only_constrained":     {"retention": False, "prefix_caching": True,  "kv_quant": None},
    "retention_constrained":             {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                                          "use_ema": True, "use_per_tool_ema": True, "pin_min": 0.10},
    "retention_pressure_constrained":    {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                                          "use_ema": True, "use_per_tool_ema": True, "pin_min": 0.40},
    # Heavy-context variants (8192-token prompts, varied tool gaps, enforce_eager + max_num_seqs=8)
    "baseline_heavy":                    {"retention": False, "prefix_caching": False, "kv_quant": None},
    "prefix_cache_only_heavy":           {"retention": False, "prefix_caching": True,  "kv_quant": None},
    "retention_heavy":                   {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                                          "use_ema": True, "use_per_tool_ema": True, "pin_min": 0.10},
    # Filler+focal variants (Daksh's microbenchmark; ttl=20s; --num-fillers driven from CLI)
    "baseline_filler":                   {"retention": False, "prefix_caching": False, "kv_quant": None},
    "prefix_cache_only_filler":          {"retention": False, "prefix_caching": True,  "kv_quant": None},
    "retention_filler":                  {"retention": True,  "prefix_caching": True,  "kv_quant": None,
                                          "use_ema": True, "use_per_tool_ema": True, "pin_min": 0.10},
}


def check_one(config_path: Path) -> tuple[bool, str]:
    """Parse one config and return (ok, summary_line)."""
    name = config_path.stem
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        return False, f"  ❌ {name:<24} parse error: {e}"

    engine_cfg = cfg.get("engine", {})
    pc = engine_cfg.get("prefix_caching", {}).get("enabled", False)
    quant = engine_cfg.get("quantization", {}).get("kv_cache")

    retention_dict = engine_cfg.get("retention", {}) or {}
    rc = build_retention_config(retention_dict)

    expected = EXPECTED.get(name)
    if expected is None:
        return True, f"  ⚠ {name:<24} no expectations defined (added since check_ablation_configs.py was written?)"

    # Walk expectations.
    issues = []
    if expected["retention"] != (rc is not None):
        issues.append(f"retention enabled? expected={expected['retention']} got={rc is not None}")
    if expected["prefix_caching"] != pc:
        issues.append(f"prefix_caching expected={expected['prefix_caching']} got={pc}")
    if expected["kv_quant"] != quant:
        issues.append(f"kv_quant expected={expected['kv_quant']} got={quant}")

    if rc is not None:
        if "use_ema" in expected and expected["use_ema"] != rc.ttl.use_ema:
            issues.append(f"use_ema expected={expected['use_ema']} got={rc.ttl.use_ema}")
        if "use_per_tool_ema" in expected and expected["use_per_tool_ema"] != rc.ttl.use_per_tool_ema:
            issues.append(f"use_per_tool_ema expected={expected['use_per_tool_ema']} got={rc.ttl.use_per_tool_ema}")
        if "pin_min" in expected and rc.pin_manager.max_pinned_fraction < expected["pin_min"]:
            issues.append(f"max_pinned_fraction too low: {rc.pin_manager.max_pinned_fraction} < {expected['pin_min']}")
        if "pin_max" in expected and rc.pin_manager.max_pinned_fraction > expected["pin_max"]:
            issues.append(f"max_pinned_fraction too high: {rc.pin_manager.max_pinned_fraction} > {expected['pin_max']}")

    if issues:
        return False, f"  ❌ {name:<24} {' | '.join(issues)}"

    # Pretty-print actual toggles for the OK row.
    rc_summary = "—"
    if rc is not None:
        rc_summary = (
            f"use_ema={rc.ttl.use_ema}, "
            f"per_tool={rc.ttl.use_per_tool_ema}, "
            f"pin={rc.pin_manager.max_pinned_fraction}"
        )
    return True, f"  ✓ {name:<24} pc={pc!s:<5} quant={str(quant):<5} retention=({rc_summary})"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"
    config_paths = sorted(configs_dir.glob("*.yaml"))

    if not config_paths:
        print(f"[!] no YAMLs found under {configs_dir}", file=sys.stderr)
        return 1

    print("=== Ablation config sanity check ===")
    print(f"  configs dir: {configs_dir}")
    print(f"  found:       {len(config_paths)} YAMLs")
    print()

    all_ok = True
    for path in config_paths:
        ok, line = check_one(path)
        if not ok:
            all_ok = False
        print(line)

    print()
    if all_ok:
        print("  All configs parse correctly with expected toggles.")
        return 0
    print("  ⚠ One or more configs failed validation. Fix before running the matrix.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
