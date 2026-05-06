"""Workstream D — Multi-trace concurrent runner.

Drives N traces simultaneously through a single vLLM engine to expose KV
cache contention. Single-agent benchmarks (run_eval.run_real) cannot
differentiate retention from prefix-cache because there is no competing
eviction force. This runner creates that contention by interleaving
multiple agents' requests through one shared block manager.

Architecture:

  - Each trace is a _TraceState carrying its own conversation, turn pointer,
    and pending request_id.
  - The main loop progresses every trace one step at a time:
      * If the trace is between turns and not in a tool gap → submit the
        next user_prompt to the engine.
      * If the trace is mid-tool-gap → check expiry, then advance.
      * If the trace has a pending request → wait for the engine to finish.
  - Up to ``concurrency`` requests are in flight simultaneously. The engine
    batches them via its native scheduler; vLLM's preemption logic kicks
    in when KV pressure exceeds the budget.
  - For real vLLM engines (anything with ``.llm_engine``), we drive
    ``add_request`` + ``step()`` directly. For MockEngine, we fall back to
    sequential ``generate()`` calls — concurrency is structurally simulated
    rather than real, but per-trace metric attribution still works for
    unit testing.

The output shape is identical to run_eval.run_real: list[TraceResult] with
per-turn TurnMetrics, peak_vram_mb, preemption_count, etc. So
comparison_table.py and the existing make targets work unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.engine_adapter import (
    EngineProtocol,
    extract_request_metrics,
    output_token_count,
)
from evaluation.metrics import TraceResult, TurnMetrics
from evaluation.trace_loader import TraceSpec, TraceTurn


# ---------------------------------------------------------------------------
# Per-trace bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class _TraceState:
    """All mutable state for one trace as the runner progresses."""

    spec: TraceSpec
    turn_idx: int = 0
    conversation: str = ""
    last_was_tool_return: bool = False
    metrics: list[TurnMetrics] = field(default_factory=list)

    # In-flight request bookkeeping.
    pending_request_id: Optional[str] = None
    pending_arrival: float = 0.0
    pending_turn: Optional[TraceTurn] = None

    # Tool-gap bookkeeping (monotonic deadline when the gap ends).
    tool_gap_ends: Optional[float] = None

    @property
    def has_more_turns(self) -> bool:
        return self.turn_idx < len(self.spec.turns)

    @property
    def finished(self) -> bool:
        """Trace has emitted all its user-prompt metrics and has no pending state."""
        return (
            not self.has_more_turns
            and self.pending_request_id is None
            and self.tool_gap_ends is None
        )


# ---------------------------------------------------------------------------
# Helpers (mirrored from run_eval; kept inline so this module is self-contained)
# ---------------------------------------------------------------------------

def _placeholder_prompt(turn: TraceTurn, conversation: str) -> str:
    body = (" foo bar baz " * max(turn.tokens // 3, 1)).strip()
    return f"{conversation}\nUSER (turn {turn.turn_index}): {body}\nASSISTANT:"


def _make_sampling_params(turn: TraceTurn) -> Any:
    try:
        from vllm import SamplingParams  # type: ignore
        return SamplingParams(
            max_tokens=turn.expected_output_tokens,
            temperature=0.0,
        )
    except ImportError:
        return {"max_tokens": turn.expected_output_tokens, "temperature": 0.0}


def _next_tool_call(turns: list[TraceTurn], i: int) -> tuple[bool, Optional[str]]:
    nxt = turns[i + 1] if i + 1 < len(turns) else None
    if nxt is not None and nxt.kind == "tool_call":
        return True, nxt.tool_name
    return False, None


def _is_real_engine(engine: Any) -> bool:
    """Detect whether engine supports the engine.llm_engine.add_request path.

    Real vLLM ``LLM`` instances expose ``.llm_engine``. ``MockEngine`` does not.
    """
    return hasattr(engine, "llm_engine") and engine.llm_engine is not None


def _handle_finished(state: _TraceState, output: Any, completion_time: float) -> None:
    """Convert one finished RequestOutput into a TurnMetrics for the trace."""
    m = extract_request_metrics(output)
    n_out = output_token_count(output)
    elapsed_ms = (completion_time - state.pending_arrival) * 1000.0

    ttft_ms = m["ttft_ms"] if m["ttft_ms"] is not None else 0.0
    decode_ms = m["decode_ms"] if m["decode_ms"] is not None else 0.0
    tbt_ms_mean = (decode_ms / n_out) if n_out > 0 else 0.0
    tbt_ms_p99 = tbt_ms_mean  # placeholder; real per-token p99 needs streaming.

    turn = state.pending_turn
    assert turn is not None, "pending_turn must be set when handling finish"

    state.metrics.append(
        TurnMetrics(
            turn_index=turn.turn_index,
            ttft_ms=ttft_ms,
            tbt_ms_mean=tbt_ms_mean,
            tbt_ms_p99=tbt_ms_p99,
            wallclock_ms=elapsed_ms,
            prefill_recomp_ms=(elapsed_ms if state.last_was_tool_return else None),
            tool_name=turn.tool_name,
        )
    )

    # Append assistant text to running conversation.
    text = ""
    if output.outputs and getattr(output.outputs[0], "text", None):
        text = output.outputs[0].text[:200]
    state.conversation += f"\nASSISTANT: {text}"

    state.pending_request_id = None
    state.pending_turn = None
    state.last_was_tool_return = False
    state.turn_idx += 1


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_concurrent(
    engine: EngineProtocol,
    specs: list[TraceSpec],
    cfg: dict[str, Any],
    concurrency: int = 1,
    *,
    poll_seconds: float = 0.01,
) -> list[TraceResult]:
    """Drive multiple TraceSpecs through one engine with bounded concurrency.

    Args:
        engine:      Real vLLM ``LLM`` instance OR MockEngine.
        specs:       Trace specifications. Each runs to completion.
        cfg:         Parsed config dict (used for config_name + metadata).
        concurrency: Maximum number of in-flight requests at any time. The
                     engine batches them; vLLM's scheduler handles eviction.
        poll_seconds: Sleep when no progress is possible (waiting on tool gaps
                     with no in-flight engine work).
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    config_name = cfg.get("name", "unknown")
    states = [_TraceState(spec=spec) for spec in specs]

    cuda_available = False
    try:
        import torch  # type: ignore
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    real = _is_real_engine(engine)

    # ----- Main loop ------------------------------------------------------
    while not all(s.finished for s in states):
        # 1. Advance each non-pending trace as far as it can go without
        #    exceeding the concurrency budget.
        progress_made = False
        active_inflight = sum(1 for s in states if s.pending_request_id is not None)

        for state in states:
            if state.finished:
                continue
            if state.pending_request_id is not None:
                continue
            if state.tool_gap_ends is not None:
                if time.monotonic() >= state.tool_gap_ends:
                    state.tool_gap_ends = None
                    progress_made = True
                else:
                    continue
            if not state.has_more_turns:
                continue

            turn = state.spec.turns[state.turn_idx]

            if turn.kind == "user_prompt":
                # Honour concurrency budget.
                if active_inflight >= concurrency:
                    continue

                prompt = _placeholder_prompt(turn, state.conversation)
                sp = _make_sampling_params(turn)
                is_pending, next_tool = _next_tool_call(state.spec.turns, state.turn_idx)
                request_id = (
                    f"{state.spec.trace_id}-t{state.turn_idx}-{time.time_ns()}"
                )

                state.pending_request_id = request_id
                state.pending_arrival = time.monotonic()
                state.pending_turn = turn

                if real:
                    engine.llm_engine.add_request(  # type: ignore[attr-defined]
                        request_id,
                        prompt,
                        sp,
                        program_id=state.spec.trace_id,
                        is_tool_call_pending=is_pending,
                        tool_name=next_tool,
                    )
                    active_inflight += 1
                else:
                    # Mock path — generate synchronously, register completion now.
                    outputs = engine.generate(
                        prompt,
                        sp,
                        program_id=state.spec.trace_id,
                        is_tool_call_pending=is_pending,
                        tool_name=next_tool,
                    )
                    if outputs:
                        _handle_finished(state, outputs[0], time.monotonic())
                progress_made = True

            elif turn.kind == "tool_call":
                if turn.tool_latency_ms:
                    state.tool_gap_ends = time.monotonic() + (turn.tool_latency_ms / 1000.0)
                state.turn_idx += 1
                progress_made = True

            elif turn.kind == "tool_return":
                state.conversation += (
                    f"\nTOOL_RETURN ({turn.tool_name or 'unknown'}): [stub output]"
                )
                state.last_was_tool_return = True
                state.turn_idx += 1
                progress_made = True

        # 2. Drive engine forward (real path only).
        if real:
            inflight = sum(1 for s in states if s.pending_request_id is not None)
            if inflight > 0:
                step_outputs = engine.llm_engine.step()  # type: ignore[attr-defined]
                completion_time = time.monotonic()
                for output in step_outputs:
                    if not getattr(output, "finished", False):
                        continue
                    state = next(
                        (s for s in states if s.pending_request_id == output.request_id),
                        None,
                    )
                    if state is not None:
                        _handle_finished(state, output, completion_time)
                        progress_made = True

        # 3. If we made no progress, sleep briefly to let tool gaps elapse.
        if not progress_made:
            in_gap = [s for s in states if s.tool_gap_ends is not None]
            if in_gap:
                next_wake = min(s.tool_gap_ends for s in in_gap)
                sleep = max(0.0, next_wake - time.monotonic())
                time.sleep(min(sleep, poll_seconds * 10))
            else:
                # No progress, no gaps — should be impossible if `finished` is correct.
                # Safety break.
                break

    # ----- Build TraceResults --------------------------------------------
    results: list[TraceResult] = []
    for state in states:
        tr = TraceResult(config_name=config_name, trace_id=state.spec.trace_id)
        tr.metadata["model"] = state.spec.model
        tr.metadata["tool_latency_dist"] = state.spec.tool_latency_dist
        tr.metadata["concurrency"] = concurrency
        tr.metadata["num_concurrent_specs"] = len(specs)
        tr.turns = state.metrics
        results.append(tr)

    if cuda_available:
        try:
            import torch  # type: ignore
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            for tr in results:
                tr.peak_vram_mb = peak
        except Exception:  # pragma: no cover — defensive
            pass

    if real:
        llm_engine = getattr(engine, "llm_engine", None)
        if llm_engine is not None:
            schedulers = getattr(llm_engine, "scheduler", None) or []
            if not isinstance(schedulers, list):
                schedulers = [schedulers]
            total_preempt = sum(
                getattr(s, "num_cumulative_preemption", 0) for s in schedulers
            )
            for tr in results:
                tr.preemption_count = int(total_preempt)

    return results
