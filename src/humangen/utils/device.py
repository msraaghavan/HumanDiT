"""Device and numeric-precision policy.

Measured on the GTX 1650 Ti (sm_75): fp16 GEMM is 4x *slower* than fp32 and bf16 is emulated,
so pre-Ampere GPUs run in fp32. Ampere+ (sm >= 80) uses bf16 autocast with TF32 matmuls.
TPUs (torch_xla, opted in with PJRT_DEVICE=TPU or HUMANGEN_DEVICE=xla) use bf16 autocast.

TPU support is UNVERIFIED: its first real run (Colab v6e-1) hit two known bugs that were deliberately left unfixed
(SDPA input dtypes under XLA autocast; torch_xla.launch on a single chip) -- docs/compute_platforms.md, section 3.
"""

from __future__ import annotations

import contextlib
import os
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


def xla_device() -> torch.device | None:
    """The TPU, if torch_xla is installed and a TPU runtime was requested; otherwise None."""
    wanted = os.environ.get("HUMANGEN_DEVICE", "").lower() == "xla" or os.environ.get("PJRT_DEVICE") == "TPU"
    if not wanted:
        return None
    try:
        import torch_xla
    except ImportError:
        return None
    return torch_xla.device()


def get_precision(force: str | None = None) -> Precision:
    """`force` ("fp32" / "bf16", or the HUMANGEN_PRECISION environment variable) skips the capability check. That
    matters on Hugging Face ZeroGPU, where no real GPU is attached until a job runs."""
    force = force or os.environ.get("HUMANGEN_PRECISION") or None
    xdev = xla_device()
    if xdev is not None:
        return Precision(xdev, None if force == "fp32" else torch.bfloat16)
    if not torch.cuda.is_available():
        return Precision(torch.device("cpu"), None)
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if force == "fp32":
        return Precision(dev, None)
    if force == "bf16" or torch.cuda.get_device_capability(dev)[0] >= 8:
        return Precision(dev, torch.bfloat16)
    return Precision(dev, None)


def full_precision_matmuls(device: torch.device) -> None:
    """TPUs run fp32 matmuls as one bf16 pass by default; request true fp32 (6-pass) accuracy. The latent
    encoder needs it: the RAEv2 latent statistics were fit on full-precision features."""
    if device.type != "xla":
        return
    torch.set_float32_matmul_precision("highest")
    try:
        import torch_xla.backends

        torch_xla.backends.set_mat_mul_precision("highest")
    except (ImportError, AttributeError):  # older torch_xla
        import torch_xla

        torch_xla._XLAC._xla_set_mat_mul_precision("highest")


def world() -> tuple[int, int]:
    """(rank, world size) of an XLA data-parallel run (one process per TPU chip); (0, 1) everywhere else."""
    if xla_device() is None:
        return 0, 1
    import torch_xla.runtime as xr

    return xr.global_ordinal(), xr.world_size()


def average_gradients(params, device: torch.device) -> None:
    """Data parallel: replace each gradient by its mean over TPU replicas. No-op on a single device."""
    if device.type != "xla":
        return
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    if xr.world_size() > 1:
        grads = [p.grad for p in params if p.grad is not None]
        xm.all_reduce(xm.REDUCE_SUM, grads, scale=1.0 / xr.world_size())


def sync(device: torch.device) -> None:
    """Finish queued work. CUDA: wait for kernels. XLA: cut the lazily traced graph here and run it, so loops
    (ODE steps, optimizer steps) compile one small reusable graph instead of one giant unrolled one."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xla":
        import torch_xla

        torch_xla.sync()
