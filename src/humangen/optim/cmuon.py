"""CMuon: Muon with chunked orthogonalisation (arXiv:2608.02502), using Polar-Express Newton–Schulz.

Plain Muon orthogonalises each 2D weight's momentum as one matrix. For *fused* weights (for example
an adaLN projection producing shift/scale/gate for attention and MLP) this couples semantically
independent sub-matrices. CMuon splits such weights along their longer dimension into
`chunks` pieces and orthogonalises each piece separately.

Update for each (chunk of a) matrix W ∈ R^{m×n}:
    B ← μB + G;   U = G + μB (Nesterov);   O = polar(U);   W ← W − lr · 0.2·√max(m, n) · O
The 0.2·√max(m, n) factor matches the update RMS of AdamW, so AdamW learning rates carry over
(this is RAEv2's `adjust_lr="rms_norm"`).
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer

# Polar Express coefficients (Amsel et al., ICLR 2026) with the 1.01 safety factor folded in,
# as published in NVIDIA's emerging-optimizers `muon_utils` ("polar_express" preset).
POLAR_EXPRESS = (
    (8.2051, -22.9019, 16.4607),
    (4.0664, -2.8612, 0.5184),
    (3.9096, -2.8234, 0.5250),
    (3.2856, -2.4153, 0.4853),
    (2.2779, -1.6198, 0.3985),
    (1.8726, -1.2307, 0.3585),
    (1.8564, -1.2132, 0.3568),
    (1.8750, -1.2500, 0.3750),
)


def polar_express(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor UVᵀ of g (..., m, n) with quintic iterations."""
    dtype = (
        torch.bfloat16 if g.is_cuda and torch.cuda.get_device_capability(g.device)[0] >= 8 else torch.float32
    )
    x = g.to(dtype)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    for i in range(steps):
        a, b, c = POLAR_EXPRESS[min(i, len(POLAR_EXPRESS) - 1)]
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


def orthogonalized_update(u: torch.Tensor, chunks: int, steps: int) -> torch.Tensor:
    """Chunk along the longer dim, orthogonalise each chunk, and apply the RMS-matching scale."""
    m, n = u.shape
    if chunks <= 1:
        return polar_express(u, steps) * (0.2 * max(m, n) ** 0.5)
    dim = 0 if m >= n else 1
    assert u.shape[dim] % chunks == 0, f"dim {u.shape[dim]} not divisible into {chunks} chunks"
    parts = u.chunk(chunks, dim=dim)
    pm, pn = parts[0].shape
    # equal-size chunks -> orthogonalise as one batched matmul
    o = polar_express(torch.stack(parts), steps) * (0.2 * max(pm, pn) ** 0.5)
    return torch.cat(o.unbind(0), dim=dim)


class CMuon(Optimizer):
    """Chunked Muon for 2D weights. Each param group may set `chunks` (default 1 = plain Muon)."""

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        chunks: int = 1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            chunks=chunks,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"CMuon only handles 2D weights, got shape {tuple(p.shape)}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu, lr, wd = group["momentum"], group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)
                u = g.add(buf, alpha=mu) if group["nesterov"] else buf
                upd = orthogonalized_update(u, group["chunks"], group["ns_steps"])
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(upd.to(p.dtype), alpha=-lr)
        return loss
