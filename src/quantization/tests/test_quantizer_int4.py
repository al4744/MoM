from __future__ import annotations

import torch

from src.quantization.quantizer import dequantize_int4, quantize_int4


def test_int4_round_trip_shape_and_dtype() -> None:
    x = torch.randn(3, 7, dtype=torch.float16)
    q = quantize_int4(x)
    y = dequantize_int4(q)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_int4_round_trip_error_bound() -> None:
    x = torch.randn(64, dtype=torch.float32)
    y = dequantize_int4(quantize_int4(x))
    mae = (x - y).abs().mean().item()
    assert mae < 0.2
