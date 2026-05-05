from __future__ import annotations

import torch

from src.quantization.types import QuantizedBlock


def _symmetric_scale(t: torch.Tensor, qmax: float) -> torch.Tensor:
    max_abs = t.abs().amax()
    scale = max_abs / qmax
    if torch.count_nonzero(scale) == 0:
        return torch.tensor(1.0, device=t.device, dtype=torch.float32)
    return scale.to(torch.float32)


def quantize_int8(t: torch.Tensor) -> QuantizedBlock:
    x = t.detach().to(torch.float32)
    scale = _symmetric_scale(x, 127.0)
    q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
    return QuantizedBlock(
        payload=q,
        scale=scale,
        zero_point=None,
        orig_shape=tuple(t.shape),
        orig_dtype=t.dtype,
        scheme="int8_symmetric",
        device=str(t.device),
    )


def dequantize_int8(q: QuantizedBlock) -> torch.Tensor:
    x = q.payload.to(torch.float32) * q.scale
    return x.to(dtype=q.orig_dtype).view(q.orig_shape)


def _pack_int4(values_0_15: torch.Tensor) -> torch.Tensor:
    if values_0_15.numel() % 2 != 0:
        values_0_15 = torch.cat(
            [values_0_15, torch.zeros(1, dtype=values_0_15.dtype, device=values_0_15.device)]
        )
    even = values_0_15[0::2]
    odd = values_0_15[1::2]
    return (even | (odd << 4)).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor, original_numel: int) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = low
    out[1::2] = high
    return out[:original_numel]


def quantize_int4(t: torch.Tensor) -> QuantizedBlock:
    x = t.detach().to(torch.float32).reshape(-1)
    scale = _symmetric_scale(x, 7.0)
    q_signed = torch.clamp(torch.round(x / scale), -7, 7).to(torch.int16)
    q_shifted = (q_signed + 8).to(torch.uint8)  # map [-7,7] -> [1,15]
    payload = _pack_int4(q_shifted)
    return QuantizedBlock(
        payload=payload,
        scale=scale,
        zero_point=torch.tensor(8, device=payload.device, dtype=torch.int16),
        orig_shape=tuple(t.shape),
        orig_dtype=t.dtype,
        scheme="int4_symmetric",
        device=str(t.device),
    )


def dequantize_int4(q: QuantizedBlock) -> torch.Tensor:
    numel = 1
    for d in q.orig_shape:
        numel *= d
    unpacked = _unpack_int4(q.payload.to(torch.uint8), numel).to(torch.int16)
    signed = unpacked - 8
    x = signed.to(torch.float32) * q.scale
    return x.to(dtype=q.orig_dtype).view(q.orig_shape)
