"""Workstream D — Synthetic trace specification loader.

Decouples D from C: until benchmarks/trace_generator.py exists, D can load
trace specs from a JSON file and pass them through to the engine. This is the
contract surface — when C lands their generator, it must emit JSON in this
same shape.

Schema:
    {
      "trace_id": "50turn-mixed",
      "model": "meta-llama/Meta-Llama-3-8B",
      "prompt_tokens": 1024,
      "tool_latency_dist": "mixed",
      "turns": [
        {
          "turn_index": 0,
          "kind": "user_prompt",                       # user_prompt | tool_call | tool_return
          "tokens": 512,
          "tool_name": null,                           # populated for tool_call/tool_return
          "tool_latency_ms": null,                     # populated for tool_call
          "expected_output_tokens": 64
        },
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Schema dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TraceTurn:
    turn_index: int
    kind: str  # user_prompt | tool_call | tool_return
    tokens: int
    tool_name: Optional[str] = None
    tool_latency_ms: Optional[float] = None
    expected_output_tokens: int = 64

    @property
    def is_tool_call(self) -> bool:
        return self.kind == "tool_call"

    @property
    def is_tool_return(self) -> bool:
        return self.kind == "tool_return"


@dataclass
class TraceSpec:
    """One synthetic conversation trace specification.

    This is the contract Workstream C's trace_generator.py must emit. Each
    TraceSpec is consumed by run_eval.run_real() to drive the engine.

    ``role`` controls how concurrent_runner treats this trace:
      * ``"focal"``  — measured trace whose latency is the headline. Standard
                       behaviour: runs to completion, tool gaps respected.
      * ``"filler"`` — background-pressure trace. Submits requests continuously
                       (no tool gaps) to fill the prefix cache and force
                       eviction. concurrent_runner stops submitting filler
                       requests once all focal traces have completed, then
                       drains in-flight requests so partial filler metrics are
                       still recorded but the run does not idle waiting on
                       fillers that have many turns left.
    """

    trace_id: str
    model: str
    prompt_tokens: int
    tool_latency_dist: str
    turns: list[TraceTurn] = field(default_factory=list)
    role: str = "focal"  # "focal" | "filler"

    # Wall-clock delay (in ms) from the start of the run before this trace
    # submits its first request. Used for the staggered / burst workload
    # classes to model agent arrivals over time. 0.0 = start immediately.
    start_offset_ms: float = 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)

    @property
    def num_tool_calls(self) -> int:
        return sum(1 for t in self.turns if t.is_tool_call)

    @property
    def num_post_tool_turns(self) -> int:
        """Turns that come immediately after a tool_return (= post-tool prefill turns)."""
        count = 0
        for prev, cur in zip(self.turns, self.turns[1:]):
            if prev.is_tool_return and cur.kind == "user_prompt":
                count += 1
        return count


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_trace(path: str | Path) -> TraceSpec:
    """Load a single TraceSpec from a JSON file."""
    with open(path) as f:
        blob = json.load(f)
    turns = [TraceTurn(**t) for t in blob.pop("turns", [])]
    return TraceSpec(turns=turns, **blob)


def load_trace_directory(path: str | Path) -> list[TraceSpec]:
    """Load every *.json TraceSpec from a directory, sorted by filename.

    Used by run_eval.run_real() once C's trace generator emits files into
    a known directory.
    """
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(f"{p} is not a directory")
    return [load_trace(child) for child in sorted(p.glob("*.json"))]


# ---------------------------------------------------------------------------
# Lightweight in-memory generator (until C lands its real generator)
# ---------------------------------------------------------------------------

def fixture_trace(
    trace_id: str = "10turn-fixture",
    num_turns: int = 10,
    tool_every: int = 3,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 1024,
    tool_latency_ms: float = 0.0,
    role: str = "focal",
) -> TraceSpec:
    """Build a deterministic in-memory TraceSpec for tests.

    Pattern: user_prompt → (every tool_every-th turn) tool_call + tool_return →
    repeat. Useful for exercising D's pipeline without writing files to disk.

    ``role`` is forwarded onto the resulting TraceSpec — defaults to
    ``"focal"`` so existing callers (and 19 prior tests) see no change.
    """
    turns: list[TraceTurn] = []
    idx = 0
    while idx < num_turns:
        turns.append(
            TraceTurn(
                turn_index=idx,
                kind="user_prompt",
                tokens=prompt_tokens // num_turns,
                expected_output_tokens=64,
            )
        )
        idx += 1
        if idx < num_turns and idx % tool_every == 0:
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
            if idx < num_turns:
                turns.append(
                    TraceTurn(
                        turn_index=idx,
                        kind="tool_return",
                        tokens=128,
                        tool_name="search",
                    )
                )
                idx += 1

    return TraceSpec(
        trace_id=trace_id,
        model=model,
        prompt_tokens=prompt_tokens,
        tool_latency_dist="fixture-fixed",
        turns=turns[:num_turns],
        role=role,
    )


# ---------------------------------------------------------------------------
# Filler+focal workload (Daksh's microbenchmark pattern)
# ---------------------------------------------------------------------------

def _focal_trace(
    *,
    trace_id: str,
    num_user_prompts: int,
    tool_latency_ms: float,
    model: str,
    prompt_tokens: int,
    body_tokens_per_turn: int,
    expected_output_tokens: int,
) -> TraceSpec:
    """Hand-crafted focal trace: u → (tc → tr → u) × N, each user_prompt
    followed by a tool gap.

    Unlike ``fixture_trace`` (which divides ``prompt_tokens`` across N turns),
    here every user_prompt carries a uniform ``body_tokens_per_turn`` so each
    focal turn occupies a fixed amount of cache regardless of N.
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
        # Tool gap follows every user_prompt EXCEPT the last (no point in
        # gating after the final turn — nothing comes after).
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
        tool_latency_dist="filler-focal-fixed",
        turns=turns,
        role="focal",
    )


def _filler_trace(
    *,
    trace_id: str,
    num_turns: int,
    model: str,
    prompt_tokens: int,
    body_tokens_per_turn: int,
    expected_output_tokens: int,
) -> TraceSpec:
    """Hand-crafted filler trace: ``num_turns`` user_prompts, no tool gaps.

    Filler purpose is to keep submitting requests continuously while focal
    traces are paused mid-tool-gap, generating cache pressure that evicts
    the focal trace's blocks unless retention pins them.
    """
    turns = [
        TraceTurn(
            turn_index=i,
            kind="user_prompt",
            tokens=body_tokens_per_turn,
            expected_output_tokens=expected_output_tokens,
        )
        for i in range(num_turns)
    ]
    return TraceSpec(
        trace_id=trace_id,
        model=model,
        prompt_tokens=prompt_tokens,
        tool_latency_dist="filler-no-gaps",
        turns=turns,
        role="filler",
    )


def filler_focal_workload(
    *,
    num_fillers: int,
    focal_num_user_prompts: int = 4,
    focal_tool_latency_ms: float = 5000.0,
    filler_num_turns: int = 200,
    model: str = "meta-llama/Meta-Llama-3-8B",
    prompt_tokens: int = 8192,
    focal_body_tokens_per_turn: int = 2048,
    filler_body_tokens_per_turn: int = 2048,
    expected_output_tokens: int = 64,
) -> list[TraceSpec]:
    """Build the (1 focal × N fillers) workload Daksh used to demonstrate
    retention's 7.9× speedup.

    Defaults reflect the heavy-context regime:
      * focal: 4 user_prompts × 2048 body tokens, 5-second tool gaps
      * filler: 200 user_prompts × 2048 body tokens, no tool gaps

    The scheduler in concurrent_runner exits when all focal traces complete,
    so ``filler_num_turns`` only needs to exceed what fillers can chew through
    in (focal_num_user_prompts × focal_tool_latency_ms / per_turn_ms) seconds.
    200 is a generous ceiling that won't run out before focal finishes even at
    short tool gaps.
    """
    if num_fillers < 0:
        raise ValueError(f"num_fillers must be >= 0, got {num_fillers}")
    if focal_num_user_prompts < 1:
        raise ValueError(
            f"focal_num_user_prompts must be >= 1, got {focal_num_user_prompts}"
        )

    focal = _focal_trace(
        trace_id="focal-0",
        num_user_prompts=focal_num_user_prompts,
        tool_latency_ms=focal_tool_latency_ms,
        model=model,
        prompt_tokens=prompt_tokens,
        body_tokens_per_turn=focal_body_tokens_per_turn,
        expected_output_tokens=expected_output_tokens,
    )
    fillers = [
        _filler_trace(
            trace_id=f"filler-{i}",
            num_turns=filler_num_turns,
            model=model,
            prompt_tokens=prompt_tokens,
            body_tokens_per_turn=filler_body_tokens_per_turn,
            expected_output_tokens=expected_output_tokens,
        )
        for i in range(num_fillers)
    ]
    return [focal] + fillers
