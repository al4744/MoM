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
import os
import sys
import time
import uuid
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
from src.quantization.config import load_kv_quant_config


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def _default_wandb_mode() -> str:
    mode = os.environ.get("WANDB_MODE", "online")
    return mode if mode in {"online", "offline", "disabled"} else "online"


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
        help="Log evaluation config, metrics, tables, and output artifacts to WandB.",
    )
    p.add_argument(
        "--wandb-project",
        default="mom-eval",
        help="WandB project name. Defaults to mom-eval.",
    )
    p.add_argument(
        "--wandb-entity",
        default=None,
        help="Optional WandB entity/team.",
    )
    p.add_argument(
        "--wandb-group",
        default=None,
        help="Optional WandB run group for baseline/optimized comparisons.",
    )
    p.add_argument(
        "--wandb-tags",
        default="",
        help="Comma-separated WandB tags. Config name and execution mode are added automatically.",
    )
    p.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=_default_wandb_mode(),
        help="WandB mode. Defaults to WANDB_MODE if set, otherwise online.",
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


def _configure_retention_events(output_dir: Path, *, use_wandb: bool) -> None:
    """Point Daksh's structured event logger at output_dir/events.jsonl.

    Soft-imports the retention package so this function works even when the
    vLLM-side retention modules are unavailable (e.g. during pure --dry-run
    on a machine without the project root checked out). Failures here are
    deliberately swallowed; an absent events.jsonl is informative on its own.
    """
    try:
        from src.retention import events
    except ImportError:
        return
    try:
        events.configure(
            log_file=str(output_dir / "events.jsonl"),
            use_wandb=use_wandb,
        )
    except Exception:  # pragma: no cover — defensive
        pass


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

def _compression_ratio_for_mode(mode: str | None) -> float | None:
    if mode == "int8":
        return 0.50
    if mode == "int4":
        return 0.25
    return None


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

    # Stable program_id for the whole trace — identifies this agent session to
    # the retention system so KV blocks can be pinned across tool-call turns.
    program_id = str(uuid.uuid4())

    # Precompute which turn indices are immediately followed by a tool_call so
    # we can set is_tool_call_pending=True on the preceding user_prompt.
    tool_call_after: set[int] = {
        spec.turns[i].turn_index
        for i in range(len(spec.turns) - 1)
        if spec.turns[i + 1].kind == "tool_call"
    }

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
        is_tool_call_pending = turn.turn_index in tool_call_after
        # When a tool call is pending, look up the tool name from the next turn.
        pending_tool_name: Optional[str] = None
        if is_tool_call_pending:
            idx = next(
                j for j, t in enumerate(spec.turns)
                if t.turn_index == turn.turn_index
            )
            if idx + 1 < len(spec.turns):
                pending_tool_name = spec.turns[idx + 1].tool_name

        prompt = _placeholder_prompt(turn, conversation)
        sp = _make_sampling_params(turn)

        with TimingContext() as turn_timer:
            import inspect as _inspect
            _gen_sig = _inspect.signature(engine.generate)
            _extra = {}
            if "program_id" in _gen_sig.parameters:
                _extra["program_id"] = program_id
                _extra["is_tool_call_pending"] = is_tool_call_pending
                _extra["tool_name"] = pending_tool_name
            outputs = engine.generate(
                prompt,
                sp,
                use_tqdm=False,
                **_extra,
            )

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
    # llm_engine.scheduler is a List[Scheduler] (one per pipeline stage);
    # sum across all stages and use num_cumulative_preemption (the real attr).
    llm_engine = getattr(engine, "llm_engine", None)
    if llm_engine is not None:
        schedulers = getattr(llm_engine, "scheduler", None) or []
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        trace.preemption_count = sum(
            getattr(s, "num_cumulative_preemption", 0) for s in schedulers
        )

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

    traces = [run_trace(engine, spec, cfg) for spec in specs]

    # Annotate Workstream B quantization metrics from config.
    quant_cfg = load_kv_quant_config(cfg.get("engine", {}).get("quantization"))
    if quant_cfg.enabled:
        ratio = _compression_ratio_for_mode(quant_cfg.mode)
        for trace in traces:
            if ratio is not None and trace.peak_vram_mb and trace.peak_vram_mb > 0:
                trace.quantized_kv_mb = trace.peak_vram_mb * ratio
                trace.kv_compression_ratio = ratio
                trace.metadata["kv_quant_mode"] = quant_cfg.mode

    return traces


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

    # Subscribe Daksh's retention event log to a per-run JSONL file.
    # No-op for dry-run / mock modes since no PinManager is active, but
    # always wire it so a file is created (an empty file is meaningful — it
    # confirms retention was either disabled or genuinely silent for this run).
    _configure_retention_events(args.output, use_wandb=args.wandb)

    if args.dry_run:
        execution_mode = "dry-run"
        traces = [
            stub_trace(config_name, ts["id"], ts.get("turns", 10))
            for ts in trace_specs
        ]
    elif args.mock_engine:
        execution_mode = "mock"
        from evaluation.engine_adapter import MockEngine
        traces = run_real(cfg, engine=MockEngine())
    else:
        execution_mode = "real"
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
        from evaluation.wandb_logger import WandbSettings, log_evaluation_run

        tags = tuple(tag.strip() for tag in args.wandb_tags.split(",") if tag.strip())
        try:
            log_evaluation_run(
                settings=WandbSettings(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    group=args.wandb_group,
                    tags=tags,
                    mode=args.wandb_mode,
                ),
                cfg=cfg,
                config_path=args.config,
                output_dir=args.output,
                traces=traces,
                summary=summary,
                execution_mode=execution_mode,
            )
        except Exception as e:  # pragma: no cover - optional integration guard
            print(
                f"[!] wandb logging failed; continuing without WandB: {e}",
                file=sys.stderr,
            )

    print(f"\n  config={config_name} traces={len(traces)} ttft_ms={summary.mean_ttft_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
