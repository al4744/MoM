from __future__ import annotations


def should_quantize_turn(turn_index: int, latest_turn_index: int, recent_turns_fp: int) -> bool:
    """Return whether a turn is old enough to quantize."""
    return (latest_turn_index - turn_index) >= recent_turns_fp
