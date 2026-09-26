"""Internal Guidance (IG) and attribute classifier-free guidance, as in RAEv2's guidance utilities.

IG:  x̂ = x̂_base + w_ig * (x̂_full - x̂_base)          applied for t in [ig_t_min, ig_t_max]
CFG: x̂ = x̂_u + w_cfg * (x̂_c - x̂_u)                  applied for t in [cfg_t_min, cfg_t_max]
(IG is applied to both the conditional and unconditional branches before CFG.)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from humangen.models.layers import AttributeTokens


@dataclass
class GuidanceConfig:
    ig_scale: float = 1.78
    ig_t_min: float = 0.10
    ig_t_max: float = 1.0
    cfg_scale: float = 1.0  # 1.0 = off
    cfg_t_min: float = 0.0
    cfg_t_max: float = 0.7


def _in(t: torch.Tensor, lo: float, hi: float, like: torch.Tensor) -> torch.Tensor:
    return ((t >= lo) & (t <= hi)).view(-1, *([1] * (like.ndim - 1)))


def guided_model_fn(model: nn.Module, attrs: torch.Tensor, g: GuidanceConfig):
    """Returns f(x_t, t) -> guided x̂ for the ODE sampler."""
    use_cfg = g.cfg_scale != 1.0 and bool((attrs != AttributeTokens.NULL).any())
    null = torch.full_like(attrs, AttributeTokens.NULL)

    def ig(x_full: torch.Tensor, x_base: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if g.ig_scale == 1.0:
            return x_full
        return torch.where(
            _in(t, g.ig_t_min, g.ig_t_max, x_full), x_base + g.ig_scale * (x_full - x_base), x_full
        )

    def fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not use_cfg:
            return ig(*model(x, t, attrs), t).float()
        xf, xb = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([attrs, null]))
        xc, xu = ig(xf, xb, torch.cat([t, t])).float().chunk(2)
        return torch.where(_in(t, g.cfg_t_min, g.cfg_t_max, xc), xu + g.cfg_scale * (xc - xu), xc)

    return fn
