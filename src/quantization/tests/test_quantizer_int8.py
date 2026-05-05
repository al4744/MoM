from __future__ import annotations

import torch

from src.quantization.quantizer import dequantize_int8, quantize_int8


def test_int8_round_trip_shape_and_dtype() -> None:
    x = torch.randn(4, 8, dtype=torch.float16)
    q = quantize_int8(x)
    y = dequantize_int8(q)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_int8_round_trip_error_bound() -> None:
    x = torch.randn(32, 32, dtype=torch.float32)
    y = dequantize_int8(quantize_int8(x))
    mae = (x - y).abs().mean().item()
    assert mae < 0.03
