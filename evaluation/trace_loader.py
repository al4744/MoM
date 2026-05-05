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
    """

    trace_id: str
    model: str
    prompt_tokens: int
    tool_latency_dist: str
    turns: list[TraceTurn] = field(default_factory=list)

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
) -> TraceSpec:
    """Build a deterministic in-memory TraceSpec for tests.

    Pattern: user_prompt → (every tool_every-th turn) tool_call + tool_return →
    repeat. Useful for exercising D's pipeline without writing files to disk.
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
    )
