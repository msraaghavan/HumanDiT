# OT drifting loss follows the official W-Flow implementation (https://github.com/hanjq17/W-Flow,
# drift_loss_ot.py, MIT License, (c) 2026 Jiaqi Han): debiased entropic-OT barycentric velocity,
# feature-scale normalisation, multi-temperature aggregation with per-temperature RMS normalisation.
"""W-Flow one-step objective, applied in DINOv3 (RAEv2 latent) feature space."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_plan(
    x: torch.Tensor, y: torch.Tensor, reg: float, iters: int, diag_mask: bool, quadratic: bool = True
) -> torch.Tensor:
    """Balanced log-domain Sinkhorn, batched over dim 0. x [B,N,D], y [B,M,D] -> plan [B,N,M]."""
    b, n, _ = x.shape
    m = y.shape[1]
    c = torch.cdist(x, y)
    if quadratic:
        c = 0.5 * c * c
    if diag_mask:  # exclude self-matches when y is the generated batch itself
        d = torch.arange(min(n, m), device=x.device)
        c[:, d, d] += 1e6
    logk = -c / reg
    log_a = torch.full((b, n), -math.log(n), device=x.device, dtype=x.dtype)
    log_b = torch.full((b, m), -math.log(m), device=x.device, dtype=x.dtype)
    log_u, log_v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(iters):
        log_u = log_a - torch.logsumexp(logk + log_v[:, None, :], dim=-1)
        log_v = log_b - torch.logsumexp(logk.transpose(1, 2) + log_u[:, None, :], dim=-1)
    return torch.exp(log_u[:, :, None] + logk + log_v[:, None, :])


@torch.no_grad()
def barycentric_map(
    x: torch.Tensor, y: torch.Tensor, reg: float, iters: int, diag_mask: bool
) -> torch.Tensor:
    pi = sinkhorn_plan(x, y, reg, iters, diag_mask)
    return (pi @ y) / pi.sum(-1, keepdim=True).clamp_min(1e-12)


def wflow_loss(
    gen: torch.Tensor,
    pos: torch.Tensor,
    r_list: Iterable[float] = (0.02, 0.05, 0.2),
    iters: int = 20,
) -> tuple[torch.Tensor, dict[str, float]]:
    """gen [B,N,D] (requires grad), pos [B,M,D] real features, per independent OT group B.

    V = T_{q->p}(x) - T_{q->q}(x) (self-map with the diagonal masked, i.e. the debiased estimator),
    summed over temperatures after RMS normalisation; loss = ||x - sg(x + V)||^2 in scaled units.
    """
    gen = gen.float()
    pos = pos.float()
    s = gen.shape[-1]
    old = gen.detach()
    with torch.no_grad():
        targets = torch.cat([old, pos], dim=1)
        scale = torch.cdist(old, targets).pow(2).mean().sqrt()
        scale_in = torch.clamp(scale / s**0.5, min=1e-3)  # mean pairwise sq. distance -> ~S
        old_s, pos_s = old / scale_in, pos / scale_in
        v_agg = torch.zeros_like(old_s)
        info = {}
        for r in r_list:
            reg = float(r) * s  # regularisation scales with the quadratic cost magnitude
            v = barycentric_map(old_s, pos_s, reg, iters, False) - barycentric_map(
                old_s, old_s, reg, iters, True
            )
            f = v.pow(2).mean()
            info[f"v_rms_{r}"] = float(f.sqrt())
            v_agg += v / f.sqrt().clamp_min(1e-4)
        goal = old_s + v_agg
    loss = (gen / scale_in - goal).pow(2).mean()
    return loss, info


def latent_feature_maps(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Multi-scale features of RAE latents (B,C,16,16) -> {name: (B, locations, C)}.

    The RAEv2 latent is itself a DINOv3 representation, so no extra feature network is needed.
    """
    feats = {"tokens": z.flatten(2).transpose(1, 2)}
    for k in (2, 4):
        p = F.avg_pool2d(z, k)
        feats[f"pool{k}"] = p.flatten(2).transpose(1, 2)
    feats["global_mean"] = z.mean(dim=(-2, -1)).unsqueeze(1)
    feats["global_std"] = z.flatten(2).std(dim=-1).unsqueeze(1)
    return feats


def grouped_wflow_loss(
    gen_z: torch.Tensor, pos_z: torch.Tensor, groups: int, **kw
) -> tuple[torch.Tensor, dict]:
    """gen_z [G*n, C,H,W], pos_z [G*m, C,H,W] ordered by group. Each (group, location) is its own OT problem."""
    total, info = 0.0, {}
    gf, pf = latent_feature_maps(gen_z), latent_feature_maps(pos_z)
    for name in gf:
        g, p = gf[name], pf[name]
        n, m, loc, d = g.shape[0] // groups, p.shape[0] // groups, g.shape[1], g.shape[2]
        g = g.view(groups, n, loc, d).permute(0, 2, 1, 3).reshape(groups * loc, n, d)
        p = p.view(groups, m, loc, d).permute(0, 2, 1, 3).reshape(groups * loc, m, d)
        loss, inf = wflow_loss(g, p, **kw)
        total = total + loss
        info.update({f"{name}/{k}": v for k, v in inf.items()})
        info[f"{name}/loss"] = float(loss.detach())
    return total, info
