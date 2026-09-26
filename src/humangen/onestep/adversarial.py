"""Adversarial post-training (the GAN stage) for a one-step student.

Discriminator = the frozen teacher trunk (DINOv3-latent DiT, already an expert on faces) evaluated on
re-noised real/fake latents, with light trainable heads on several trunk depths (LADD-style).
Conditioning enters through the trunk's attribute tokens, so D judges image *and* attribute fit.
Regularisation: approximated R1 (APT, Lin et al. 2025): penalise D's sensitivity to small input noise.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from humangen.diffusion.transport import Transport
from humangen.models.dit_dh import HumanDiTDH


class TrunkDiscriminator(nn.Module):
    def __init__(self, teacher: HumanDiTDH, layers: tuple[int, ...] = (3, 6, 9, 12), hidden: int = 256):
        super().__init__()
        t = copy.deepcopy(teacher)
        self.cfg = t.cfg
        self.s_embedder, self.t_embedder, self.attr_embedder = t.s_embedder, t.t_embedder, t.attr_embedder
        self.trunk, self.rope = t.trunk[: max(layers)], t.trunk_rope
        for p in self.parameters():
            p.requires_grad_(False)
        self.layers = layers
        d = self.cfg.trunk_dim
        self.heads = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            for _ in layers
        )

    def trainable_parameters(self):
        return self.heads.parameters()

    def forward(self, x: torch.Tensor, t: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        """x: latent (B,C,H,W) at noise level t -> logits (B,)."""
        n = x.shape[-2] * x.shape[-1]
        tokens = x.flatten(2).transpose(1, 2)
        _, t_tokens = self.t_embedder(t)
        seq = torch.cat([self.s_embedder(tokens), t_tokens, self.attr_embedder(attrs)], dim=1)
        out, hi = 0.0, 0
        for i, blk in enumerate(self.trunk):
            # G step: gradients flow through the frozen trunk into the student, so recompute its activations
            # instead of storing them (a 24 GB GPU cannot hold both models' activations at batch 64 in eager mode).
            if torch.is_grad_enabled() and seq.requires_grad:
                seq = checkpoint(blk, seq, self.rope, use_reentrant=False)
            else:
                seq = blk(seq, self.rope)
            if i + 1 in self.layers:
                out = out + self.heads[hi](seq[:, :n]).mean(dim=(1, 2))
                hi += 1
        return out


def renoise(x: torch.Tensor, transport: Transport, t_max: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample t ~ shifted-uniform on (0, t_max] and noise x (D sees clean-ish to fairly noisy inputs)."""
    t = transport.sample_t(x.shape[0], x.device).clamp(max=t_max)
    return transport.interpolate(x, torch.randn_like(x), t), t


def d_loss(
    d: TrunkDiscriminator,
    real: torch.Tensor,
    real_attrs,
    fake: torch.Tensor,
    fake_attrs,
    transport: Transport,
    r1_weight: float,
    r1_sigma: float,
) -> tuple[torch.Tensor, dict]:
    xr, tr = renoise(real, transport)
    xf, tf = renoise(fake.detach(), transport)
    lr_, lf = d(xr, tr, real_attrs), d(xf, tf, fake_attrs)
    loss = F.softplus(-lr_).mean() + F.softplus(lf).mean()
    info = {"d_real": float(lr_.detach().mean()), "d_fake": float(lf.detach().mean())}
    if r1_weight > 0:  # approximated R1: || D(x) - D(x + sigma*eps) ||^2 on real inputs
        pert = d(xr + r1_sigma * torch.randn_like(xr), tr, real_attrs)
        r1 = (pert - lr_).pow(2).mean()
        loss = loss + r1_weight * r1
        info["r1"] = float(r1.detach())
    return loss, info


def g_loss(d: TrunkDiscriminator, fake: torch.Tensor, fake_attrs, transport: Transport) -> torch.Tensor:
    xf, tf = renoise(fake, transport)
    return F.softplus(-d(xf, tf, fake_attrs)).mean()
