# Architecture follows RAEv2's `DiTwDDTHeadIG` (https://github.com/nanovisionx/RAEv2,
# src/stage2/models/DDT.py, CC BY-NC 4.0). The class-label embedder is replaced by per-attribute
# tokens so faces can be conditioned on any subset of attributes.
"""HumanDiT^DH: a DiT trunk plus a wide DDT head, with an Internal-Guidance base head."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from humangen.models.layers import (
    AttributeTokens,
    FinalLayer,
    FourierTimeTokens,
    HeadBlock,
    RoPE2D,
    TrunkBlock,
)


@dataclass(frozen=True)
class DiTDHConfig:
    in_channels: int = 1024  # RAEv2 DINOv3-L latent width
    input_size: int = 16  # 16x16 latent grid, patch size 1
    trunk_dim: int = 768
    trunk_depth: int = 12
    trunk_heads: int = 12
    head_dim: int = 1536  # DDT head must be wider than the latent (1024)
    head_depth: int = 2
    head_heads: int = 12
    mlp_ratio: float = 4.0
    num_time_tokens: int = 4
    num_attributes: int = 12
    num_extra_cond_tokens: int = 4
    ig_depth: int = 4  # Internal-Guidance base head after this trunk block

    @property
    def num_patches(self) -> int:
        return self.input_size**2

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    # 182.9M params (trunk = DiT-B width/depth, 2-block 1536-wide DDT head).
    "B": {},
    # 65.1M params; laptop smoke tests.
    "S": dict(trunk_dim=384, trunk_heads=6, head_dim=1024, head_heads=8),
    # tiny; unit tests only.
    "T": dict(trunk_dim=64, trunk_depth=4, trunk_heads=2, head_dim=128, head_heads=2, ig_depth=2),
}


def make_config(preset: str, **overrides) -> DiTDHConfig:
    return DiTDHConfig(**{**PRESETS[preset], **overrides})


class HumanDiTDH(nn.Module):
    """x-prediction flow model on RAE latents.

    forward(x_t, t, attrs) -> (x_full, x_base): full-depth and ig_depth-depth clean-latent predictions.
    """

    def __init__(self, cfg: DiTDHConfig):
        super().__init__()
        self.cfg = cfg
        c, dt, dh = cfg.in_channels, cfg.trunk_dim, cfg.head_dim
        self.s_embedder = nn.Linear(c, dt)
        self.x_embedder = nn.Linear(c, dh)
        self.t_embedder = FourierTimeTokens(dt, cfg.num_time_tokens)
        self.attr_embedder = AttributeTokens(dt, cfg.num_attributes, cfg.num_extra_cond_tokens)
        self.num_cond_tokens = cfg.num_time_tokens + self.attr_embedder.n_tokens

        self.trunk = nn.ModuleList(
            TrunkBlock(dt, cfg.trunk_heads, cfg.mlp_ratio) for _ in range(cfg.trunk_depth)
        )
        self.s_projector = nn.Linear(dt, dh) if dt != dh else nn.Identity()
        self.head = nn.ModuleList(HeadBlock(dh, cfg.head_heads, cfg.mlp_ratio) for _ in range(cfg.head_depth))
        self.final_layer = FinalLayer(dh, c)
        self.base_final_layer = FinalLayer(dt, c)

        self.trunk_rope = RoPE2D(dt // cfg.trunk_heads, cfg.num_patches, self.num_cond_tokens)
        self.head_rope = RoPE2D(dh // cfg.head_heads, cfg.num_patches)
        self.gradient_checkpointing = False
        self._init_weights()

    def _init_weights(self) -> None:
        def basic(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(basic)
        for mlp in (self.t_embedder.mlp,):
            nn.init.normal_(mlp[0].weight, std=0.02)
            nn.init.normal_(mlp[2].weight, std=0.02)
        for blk in self.head:
            nn.init.zeros_(blk.adaln_modulation[-1].weight)
            nn.init.zeros_(blk.adaln_modulation[-1].bias)
        for fl in (self.final_layer, self.base_final_layer):
            for lin in (fl.adaln_modulation[-1], fl.linear):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def _run(self, fn, *args):
        if self.gradient_checkpointing and self.training:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, attrs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        n = h * w
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        t_base, t_tokens = self.t_embedder(t)
        seq = torch.cat([self.s_embedder(tokens), t_tokens, self.attr_embedder(attrs)], dim=1)

        x_base = None
        for i, blk in enumerate(self.trunk):
            seq = self._run(blk, seq, self.trunk_rope)
            if i + 1 == self.cfg.ig_depth:
                x_base = seq[:, :n]
        assert x_base is not None

        cond = self.s_projector(F.silu(t_base + seq[:, :n]))
        y = self.x_embedder(tokens)
        for blk in self.head:
            y = self._run(blk, y, cond, self.head_rope)
        x_full = self.final_layer(y, cond)

        x_base = F.silu(t_base + x_base)
        x_base = self.base_final_layer(x_base, x_base)

        def to_grid(z: torch.Tensor) -> torch.Tensor:
            return z.transpose(1, 2).reshape(b, c, h, w)

        return to_grid(x_full), to_grid(x_base)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
