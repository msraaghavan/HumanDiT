"""Device and numeric-precision policy.

Measured on the GTX 1650 Ti (sm_75): fp16 GEMM is 4x *slower* than fp32 and bf16 is emulated,
so pre-Ampere GPUs run in fp32. Ampere+ (sm >= 80) uses bf16 autocast with TF32 matmuls.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Precision:
    device: torch.device
    autocast_dtype: torch.dtype | None  # None -> plain fp32

    @property
    def name(self) -> str:
        return "fp32" if self.autocast_dtype is None else str(self.autocast_dtype).removeprefix("torch.")

    def autocast(self) -> contextlib.AbstractContextManager:
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)


def get_precision(force: str | None = None) -> Precision:
    if not torch.cuda.is_available():
        return Precision(torch.device("cpu"), None)
    dev = torch.device("cuda")
    major, _ = torch.cuda.get_device_capability(dev)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if force == "fp32":
        return Precision(dev, None)
    if force == "bf16" or (force is None and major >= 8):
        return Precision(dev, torch.bfloat16)
    return Precision(dev, None)
