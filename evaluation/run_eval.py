"""Workstream D — Evaluation runner CLI.

Dispatches a benchmark run for a single (config, trace_set) pair, collects
TraceResults, writes a per-trace JSON dump, and emits a summary.

Usage:
    # Dry-run (sentinel values; works without vLLM/CUDA):
    python evaluation/run_eval.py \\
        --config configs/baseline.yaml \\
        --output results/baseline-2026-05-05/ \\
        --dry-run

    # Real run (requires vLLM + GPU):
    python evaluation/run_eval.py \\
        --config configs/baseline.yaml \\
        --output results/baseline-2026-05-05/

Three execution modes:

  - ``--dry-run``       : sentinel TraceResults, ``-1.0`` everywhere
  - default (no flag)   : real ``vllm.LLM`` via ``build_real_engine``
  - injected ``engine=``: arbitrary EngineProtocol (used by tests with MockEngine)

Module layout:

  - ``stub_trace(...)``         — sentinel TraceResults for --dry-run
  - ``run_trace(engine, spec, cfg)`` — drive ONE trace through an engine
  - ``run_real(cfg, engine=None)``    — drive every trace in cfg through engine
  - ``main()``                  — CLI entry point

Upstream interfaces consumed:

  - configs/*.yaml             → parsed via load_config()
  - evaluation/trace_loader.py → TraceSpec / fixture_trace
  - evaluation/engine_adapter  → EngineProtocol, MockEngine, build_real_engine
  - src/retention/events.py    → per-turn pin/reuse/expire log (read post-run)
  - evaluation/metrics.py      → TurnMetrics / TraceResult / RunSummary
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from evaluation.engine_adapter import (
    EngineProtocol,
    build_real_engine,
    extract_request_metrics,
    output_token_count,
)
from evaluation.metrics import RunSummary, TimingContext, TraceResult, TurnMetrics
from evaluation.trace_loader import TraceSpec, TraceTurn, fixture_trace


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the MoM evaluation suite for one configuration.",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML config file under configs/.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for per-trace JSON + summary markdown.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip engine invocation; emit stub traces (sentinel values).",
    )
    p.add_argument(
        "--mock-engine",
        action="store_true",
        help="Use the in-process MockEngine instead of real vLLM. Useful for "
             "smoke testing the pipeline on machines without CUDA.",
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help="Push aggregated RunSummary to WandB (requires WANDB_API_KEY).",
    )
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config. Soft-imports yaml."""
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            f"PyYAML required to load {path}; pip install pyyaml"
        ) from e
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Stub generator (--dry-run)
# ---------------------------------------------------------------------------

def stub_trace(config_name: str, trace_id: str, num_turns: int) -> TraceResult:
    """Generate a deterministic stub TraceResult.

    Sentinel ``-1.0`` values make dry-run output obviously distinct from real
    measurements in any downstream comparison.
    """
    trace = TraceResult(
        config_name=config_name,
        trace_id=trace_id,
        peak_vram_mb=-1.0,
        preemption_count=-1,
        metadata={"stub": True},
    )
    for i in range(num_turns):
        trace.turns.append(
            TurnMetrics(
                turn_index=i,
                ttft_ms=-1.0,
                tbt_ms_mean=-1.0,
                tbt_ms_p99=-1.0,
                wallclock_ms=-1.0,
                prefill_recomp_ms=(-1.0 if i > 0 and i % 3 == 0 else None),
                tool_name=("search" if i % 3 == 0 else None),
            )
        )
    return trace


# ---------------------------------------------------------------------------
# Real run — drives an engine through one or many TraceSpecs
# ---------------------------------------------------------------------------

def _placeholder_prompt(turn: TraceTurn, conversation_so_far: str) -> str:
    """Render a turn's contribution to the running conversation prompt.

    Real prompt content is Workstream C's responsibility (TraceTurn currently
    only carries token counts, not text). For the smoke run we synthesize a
    deterministic placeholder of approximately the right length.
    """
    body = (" foo bar baz " * max(turn.tokens // 3, 1)).strip()
    return f"{conversation_so_far}\nUSER (turn {turn.turn_index}): {body}\nASSISTANT:"


def _make_sampling_params(turn: TraceTurn) -> Any:
    """Build SamplingParams when vLLM is available; else a plain dict."""
    try:
        from vllm import SamplingParams  # type: ignore
        return SamplingParams(
            max_tokens=turn.expected_output_tokens,
            temperature=0.0,
        )
    except ImportError:
        return {"max_tokens": turn.expected_output_tokens, "temperature": 0.0}


def run_trace(
    engine: EngineProtocol,
    spec: TraceSpec,
    cfg: dict[str, Any],
) -> TraceResult:
    """Drive one TraceSpec through an engine and collect TurnMetrics.

    Pattern per turn-spec:

      - ``user_prompt`` → call engine.generate(); time via TimingContext;
        extract TTFT/TBT from RequestOutput.metrics; append TurnMetrics.
      - ``tool_call``   → simulate tool latency with time.sleep(latency_ms).
      - ``tool_return`` → append a tool-output marker into the running context.

    The post-tool prefill recomputation field is set to wallclock_ms when the
    turn immediately follows a tool_return (no profiler attribution yet —
    that's Workstream C's responsibility once they ship).

    Peak VRAM and preemption_count are filled in if torch + vLLM are available;
    otherwise left at their defaults (None / 0) so the same code path works
    against MockEngine.
    """
    config_name = cfg.get("name", "unknown")
    trace = TraceResult(config_name=config_name, trace_id=spec.trace_id)
    trace.metadata["model"] = spec.model
    trace.metadata["tool_latency_dist"] = spec.tool_latency_dist

    # Reset CUDA peak before the trace if available.
    cuda_available = False
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    conversation = ""
    last_was_tool_return = False

    for turn in spec.turns:
        if turn.kind == "tool_call":
            if turn.tool_latency_ms:
                time.sleep(turn.tool_latency_ms / 1000.0)
            continue

        if turn.kind == "tool_return":
            conversation += f"\nTOOL_RETURN ({turn.tool_name or 'unknown'}): [stub output]"
            last_was_tool_return = True
            continue

        # turn.kind == "user_prompt"
        prompt = _placeholder_prompt(turn, conversation)
        sp = _make_sampling_params(turn)

        with TimingContext() as turn_timer:
            outputs = engine.generate(prompt, sp)

        if not outputs:
            continue
        out = outputs[0]
        m = extract_request_metrics(out)
        n_out = output_token_count(out)

        ttft_ms = m["ttft_ms"] if m["ttft_ms"] is not None else 0.0
        decode_ms = m["decode_ms"] if m["decode_ms"] is not None else 0.0
        tbt_ms_mean = (decode_ms / n_out) if n_out > 0 else 0.0
        # p99 across tokens requires per-token timestamps; placeholder = mean.
        # Workstream C will replace with profiler-derived per-token timing.
        tbt_ms_p99 = tbt_ms_mean

        trace.turns.append(
            TurnMetrics(
                turn_index=turn.turn_index,
                ttft_ms=ttft_ms,
                tbt_ms_mean=tbt_ms_mean,
                tbt_ms_p99=tbt_ms_p99,
                wallclock_ms=turn_timer.elapsed_ms,
                prefill_recomp_ms=(turn_timer.elapsed_ms if last_was_tool_return else None),
                tool_name=turn.tool_name,
            )
        )

        # Append the (truncated) generated text to the running context.
        text = ""
        if out.outputs and getattr(out.outputs[0], "text", None):
            text = out.outputs[0].text[:200]
        conversation += f"\nASSISTANT: {text}"
        last_was_tool_return = False

    if cuda_available:
        try:
            import torch  # type: ignore
            trace.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:  # pragma: no cover — defensive
            pass

    # vLLM scheduler preemption counter, if exposed.
    sched = getattr(engine, "llm_engine", None)
    if sched is not None:
        scheduler = getattr(sched, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "num_preemption"):
            trace.preemption_count = int(scheduler.num_preemption)

    return trace


def run_real(
    cfg: dict[str, Any],
    engine: Optional[EngineProtocol] = None,
    specs: Optional[list[TraceSpec]] = None,
) -> list[TraceResult]:
    """Drive every TraceSpec implied by ``cfg`` through ``engine``.

    Args:
        cfg:    Parsed YAML config. ``cfg['traces']`` lists trace specs (each
                with ``id``, ``turns``, ``tool_latency_dist``, ``prompt_tokens``).
        engine: Optional injected EngineProtocol. If None, builds a real
                ``vllm.LLM`` via ``build_real_engine(cfg)``.
        specs:  Optional override of the TraceSpec list. If None, builds them
                from cfg via ``fixture_trace`` (until C's generator lands).
    """
    model_name = cfg.get("model", {}).get("name", "unknown-model")
    trace_descs = cfg.get("traces", [])

    if specs is None:
        specs = [
            fixture_trace(
                trace_id=desc["id"],
                num_turns=desc.get("turns", 10),
                model=model_name,
                prompt_tokens=desc.get("prompt_tokens", 1024),
            )
            for desc in trace_descs
        ]

    if engine is None:
        engine = build_real_engine(cfg)

    return [run_trace(engine, spec, cfg) for spec in specs]


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    config_name = cfg.get("name", args.config.stem)
    trace_specs = cfg.get("traces", [])
    if not trace_specs:
        print(f"[!] No traces defined in {args.config}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        traces = [
            stub_trace(config_name, ts["id"], ts.get("turns", 10))
            for ts in trace_specs
        ]
    elif args.mock_engine:
        from evaluation.engine_adapter import MockEngine
        traces = run_real(cfg, engine=MockEngine())
    else:
        traces = run_real(cfg)

    # Per-trace JSON dump.
    for tr in traces:
        out_path = args.output / f"{tr.trace_id}.json"
        tr.emit_json(out_path)
        print(f"  wrote {out_path}")

    # Aggregate summary.
    summary = RunSummary.from_traces(traces)
    summary_path = args.output / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.__dict__, f, indent=2)
    print(f"  wrote {summary_path}")

    # Optional WandB push.
    if args.wandb:
        try:
            import wandb  # type: ignore
            wandb.init(project="mom-eval", name=config_name, config=cfg)
            wandb.log(summary.__dict__)
            wandb.finish()
        except ImportError:
            print("[!] wandb not installed; skipping --wandb push", file=sys.stderr)

    print(f"\n  config={config_name} traces={len(traces)} ttft_ms={summary.mean_ttft_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
