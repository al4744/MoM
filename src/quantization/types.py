from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch


@dataclass
class QuantizedBlock:
    payload: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None
    orig_shape: Tuple[int, ...]
    orig_dtype: torch.dtype
    scheme: Literal["int8_symmetric", "int4_symmetric"]
    device: str
