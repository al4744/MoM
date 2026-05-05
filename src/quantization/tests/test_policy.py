from __future__ import annotations

from src.quantization.policy import should_quantize_turn


def test_recent_turns_stay_full_precision() -> None:
    assert not should_quantize_turn(turn_index=9, latest_turn_index=10, recent_turns_fp=2)
    assert not should_quantize_turn(turn_index=10, latest_turn_index=10, recent_turns_fp=2)


def test_older_turns_become_quantized() -> None:
    assert should_quantize_turn(turn_index=3, latest_turn_index=10, recent_turns_fp=2)
