"""Workstream D — Workload class builders for the comprehensive battery.

The five workload classes the evaluation matrix covers, with the dimension
each one varies:

  W1  lockstep         — N identical agents, all start at t=0, identical
                         tool gaps. Negative control: there is no cross-
                         agent contention pattern retention can exploit, so
                         (PC, retention) should tie. If retention "loses"
                         here it would indicate a regression, not a feature.
  W2  filler_focal     — 1 measured focal trace + N continuous-submission
                         filler traces. Daksh's microbenchmark regime.
                         Fillers create persistent eviction pressure during
                         focal's tool gap. Built by ``filler_focal_workload``
                         (still lives in ``evaluation.trace_loader``).
  W3  staggered        — N agents arrive Poisson(rate). Each is identical.
                         New arrivals create eviction pressure on agents
                         that are mid-tool-gap. Sustained-load regime.
  W4  heterogeneous    — N agents start together; tool latencies are drawn
                         from a log-normal distribution. Exercises the
                         per-tool-EMA path of retention's TTL predictor.
  W5  burst            — N agents all start within ``burst_duration_ms``
                         (uniform). Transient burst — cache fills quickly
                         then eviction kicks in across all agents at once.
                         Tests retention's ability to maintain priority
                         under sudden contention.

Each builder returns ``list[TraceSpec]`` ready to feed into
``evaluation.concurrent_runner.run_concurrent`` with no further surgery.

Per-trace ``role`` is set to ``"focal"`` for all classes EXCEPT
``filler_focal``; the runner's focal/filler termination logic only kicks in
when at least one trace is filler. So lockstep / staggered / heterogeneous /
burst all run to completion with the legacy "wait for everyone" behaviour.
"""
from __future__ import annotations

import random
from typing import Iterable

from evaluation.trace_loader import (
    TraceSpec,
    TraceTurn,
    fixture_trace,
    filler_focal_workload,  # re-exported for unified import
)

__all__ = [
    "filler_focal_workload",
    "lockstep_workload",
    "staggered_workload",
    "heterogeneous_workload",
    "burst_workload",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_agent_trace(
    *,
    trace_id: str,
    num_user_prompts: int,
    tool_latency_ms: float,
    body_tokens_per_turn: int,
    expected_output_tokens: int,
    model: str,
    prompt_tokens: int,
    tool_latency_dist: str,
    start_offset_ms: float = 0.0,
    role: str = "focal",
) -> TraceSpec:
    """Hand-craft a uniform agent trace: N user_prompts × (N-1) tool gaps.

    Identical structure to ``trace_loader._focal_trace`` but exposed here so
    the workload builders own their own scheduling parameters (start_offset_ms,
    tool_latency_dist label, etc.) without monkeying with trace_loader's API.
    """
    turns: list[TraceTurn] = []
    idx = 0
    for u in range(num_user_prompts):
        turns.append(
            TraceTurn(
                turn_index=idx,
                kind="user_prompt",
                tokens=body_tokens_per_turn,
                expected_output_tokens=expected_output_tokens,
            )
        )
        idx += 1
        if u < num_user_prompts - 1:
            turns.append(
                TraceTurn(
                    turn_index=idx,
                    kind="tool_call",
                    tokens=32,
                    tool_name="search",
                    tool_latency_ms=tool_latency_ms,
                )
            )
            idx += 1
            turns.append(
                TraceTurn(
                    turn_index=idx,
                    kind="tool_return",
                    tokens=64,
                    tool_name="search",
                )
            )
            idx += 1
    return TraceSpec(
        trace_id=trace_id,
        model=model,
        prompt_tokens=prompt_tokens,
        tool_latency_dist=tool_latency_dist,
        turns=turns,
        role=role,
        start_offset_ms=start_offset_ms,
    )


def _validate_common(num_agents: int, num_user_prompts: int) -> None:
    if num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {num_agents}")
    if num_user_prompts < 1:
        raise ValueError(
            f"num_user_prompts must be >= 1, got {num_user_prompts}"
        )


# ---------------------------------------------------------------------------
# W1 — lockstep (negative control)
# ---------------------------------------------------------------------------

def lockstep_workload(
    *,
    num_agents: int,
    num_user_prompts: int = 4,
    tool_latency_ms: float = 2000.0,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 8192,
    body_tokens_per_turn: int = 2048,
    expected_output_tokens: int = 64,
) -> list[TraceSpec]:
    """N identical agents, all starting at t=0 with synchronised tool gaps.

    This is the failure mode the existing 6-regime matrix accidentally fell
    into — agents all paused for tool gaps simultaneously, leaving no
    background eviction force during any agent's pause. We ship it as W1
    explicitly so the (PC, retention) tie here is a *predicted* outcome and
    not a "we forgot to test the right thing" oversight.
    """
    _validate_common(num_agents, num_user_prompts)
    return [
        _build_agent_trace(
            trace_id=f"lockstep-{i}",
            num_user_prompts=num_user_prompts,
            tool_latency_ms=tool_latency_ms,
            body_tokens_per_turn=body_tokens_per_turn,
            expected_output_tokens=expected_output_tokens,
            model=model,
            prompt_tokens=prompt_tokens,
            tool_latency_dist="lockstep",
            start_offset_ms=0.0,
            role="focal",
        )
        for i in range(num_agents)
    ]


# ---------------------------------------------------------------------------
# W3 — staggered (Poisson arrivals)
# ---------------------------------------------------------------------------

def staggered_workload(
    *,
    num_agents: int,
    arrival_rate_per_sec: float = 0.5,
    num_user_prompts: int = 4,
    tool_latency_ms: float = 2000.0,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 8192,
    body_tokens_per_turn: int = 2048,
    expected_output_tokens: int = 64,
    seed: int = 0,
) -> list[TraceSpec]:
    """N agents, identical traces, Poisson-distributed arrivals.

    Inter-arrival times are exponential with mean 1/arrival_rate_per_sec,
    accumulated to give each agent a start_offset_ms from t=0. Agent 0
    always starts at t=0 (so the first arrival is immediate); subsequent
    agents arrive at progressively later times.

    Realistic agentic regime: load builds gradually, agents enter mid-flight
    while existing agents are paused for tool gaps. New arrivals' prefill
    blocks compete for cache space with paused agents' pinned blocks.
    """
    _validate_common(num_agents, num_user_prompts)
    if arrival_rate_per_sec <= 0.0:
        raise ValueError(
            f"arrival_rate_per_sec must be > 0, got {arrival_rate_per_sec}"
        )

    rng = random.Random(seed)
    offsets_ms: list[float] = [0.0]
    for _ in range(num_agents - 1):
        # Exponential inter-arrival → cumulative sum → arrival epoch.
        gap_s = rng.expovariate(arrival_rate_per_sec)
        offsets_ms.append(offsets_ms[-1] + gap_s * 1000.0)

    return [
        _build_agent_trace(
            trace_id=f"staggered-{i}",
            num_user_prompts=num_user_prompts,
            tool_latency_ms=tool_latency_ms,
            body_tokens_per_turn=body_tokens_per_turn,
            expected_output_tokens=expected_output_tokens,
            model=model,
            prompt_tokens=prompt_tokens,
            tool_latency_dist="staggered-poisson",
            start_offset_ms=offsets_ms[i],
            role="focal",
        )
        for i in range(num_agents)
    ]


# ---------------------------------------------------------------------------
# W4 — heterogeneous (log-normal tool latencies)
# ---------------------------------------------------------------------------

def heterogeneous_workload(
    *,
    num_agents: int,
    num_user_prompts: int = 4,
    tool_latency_log_mean_ms: float = 1500.0,
    tool_latency_log_sigma: float = 0.7,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 8192,
    body_tokens_per_turn: int = 2048,
    expected_output_tokens: int = 64,
    seed: int = 0,
) -> list[TraceSpec]:
    """N agents with per-agent tool latencies drawn from log-normal.

    The mean parameter is in ms (NOT log-space). Internally converted to
    log-space mu = ln(mean) - 0.5*sigma**2 so the resulting log-normal
    samples have an arithmetic mean of ``tool_latency_log_mean_ms``.

    This is the regime where retention's per-tool-EMA path matters: when
    tool latencies vary, a single global EMA makes retention either
    over-pin (if global EMA is high but a fast tool just returned, blocks
    pinned too long → wastes cache) or under-pin (vice versa). Per-tool
    EMA tracks each tool's distribution separately.

    All agents start at t=0; the variation comes from each agent calling
    a different tool.
    """
    import math
    _validate_common(num_agents, num_user_prompts)
    if tool_latency_log_mean_ms <= 0.0:
        raise ValueError(
            f"tool_latency_log_mean_ms must be > 0, got {tool_latency_log_mean_ms}"
        )
    if tool_latency_log_sigma < 0.0:
        raise ValueError(
            f"tool_latency_log_sigma must be >= 0, got {tool_latency_log_sigma}"
        )

    rng = random.Random(seed)
    # Solve: E[X] = e^{mu + sigma^2/2} = mean → mu = ln(mean) - sigma^2/2.
    mu = math.log(tool_latency_log_mean_ms) - 0.5 * (tool_latency_log_sigma ** 2)

    specs: list[TraceSpec] = []
    for i in range(num_agents):
        # One sample per agent (every tool call within an agent uses the
        # same latency; cross-agent variation provides the heterogeneity).
        # The TTL predictor's per-tool-EMA path triggers on per-call updates;
        # within a single agent's repeated tool calls the EMA quickly
        # converges. Cross-agent variation is what stresses the global vs
        # per-tool comparison.
        sampled_ms = rng.lognormvariate(mu, tool_latency_log_sigma)
        specs.append(
            _build_agent_trace(
                trace_id=f"heterogeneous-{i}",
                num_user_prompts=num_user_prompts,
                tool_latency_ms=sampled_ms,
                body_tokens_per_turn=body_tokens_per_turn,
                expected_output_tokens=expected_output_tokens,
                model=model,
                prompt_tokens=prompt_tokens,
                tool_latency_dist="heterogeneous-lognormal",
                start_offset_ms=0.0,
                role="focal",
            )
        )
    return specs


# ---------------------------------------------------------------------------
# W5 — burst (M agents start within T seconds)
# ---------------------------------------------------------------------------

def burst_workload(
    *,
    num_agents: int,
    burst_duration_ms: float = 500.0,
    num_user_prompts: int = 4,
    tool_latency_ms: float = 2000.0,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 8192,
    body_tokens_per_turn: int = 2048,
    expected_output_tokens: int = 64,
    seed: int = 0,
) -> list[TraceSpec]:
    """N agents arriving within a tight ``burst_duration_ms`` window.

    Start offsets uniform on [0, burst_duration_ms]. burst_duration_ms=0
    is equivalent to lockstep; burst_duration_ms → ∞ approaches uniform
    inter-arrival.

    Realistic: a UI integration that hands a single complex query to an
    agentic system, which spawns many sub-agents in quick succession.
    Cache fills sharply; eviction triggers on the late-arriving agents.
    Retention's value: protect the early-arrived agents that are now in
    tool gaps from being evicted by the late-arriving prefill bursts.
    """
    _validate_common(num_agents, num_user_prompts)
    if burst_duration_ms < 0.0:
        raise ValueError(
            f"burst_duration_ms must be >= 0, got {burst_duration_ms}"
        )

    rng = random.Random(seed)
    if num_agents == 1:
        offsets_ms = [0.0]
    else:
        offsets_ms = [rng.uniform(0.0, burst_duration_ms) for _ in range(num_agents)]
        # Sort for determinism / readability — first arrival at offset 0 is
        # not enforced (rng decides). If you want exactly t=0 for one agent,
        # patch a single index.
        offsets_ms.sort()

    return [
        _build_agent_trace(
            trace_id=f"burst-{i}",
            num_user_prompts=num_user_prompts,
            tool_latency_ms=tool_latency_ms,
            body_tokens_per_turn=body_tokens_per_turn,
            expected_output_tokens=expected_output_tokens,
            model=model,
            prompt_tokens=prompt_tokens,
            tool_latency_dist="burst-uniform",
            start_offset_ms=offsets_ms[i],
            role="focal",
        )
        for i in range(num_agents)
    ]
