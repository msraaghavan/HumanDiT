# Building blocks follow RAEv2's stage-2 model (https://github.com/nanovisionx/RAEv2,
# src/stage2/models/model_utils.py, CC BY-NC 4.0): 2D RoPE with no position for condition tokens,
# qk-RMSNorm attention with separate q/k/v projections, and SwiGLU with separate gate/up weights.
"""Transformer layers shared by the HumanDiT^DH generator."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE2D(nn.Module):
    """Axial 2D rotary embedding over a square token grid; trailing condition tokens get angle 0."""

    def __init__(self, head_dim: int, num_patches: int, num_cond_tokens: int = 0, theta: float = 10000.0):
        super().__init__()
        d, side = head_dim // 2, math.isqrt(num_patches)
        freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
        base = torch.outer(torch.arange(side).float(), freqs)  # (side, d/2)
        grid = torch.cat([base[:, None].expand(-1, side, -1), base[None, :].expand(side, -1, -1)], dim=-1)
        angles = torch.cat([grid.reshape(num_patches, d), torch.zeros(num_cond_tokens, d)], dim=0)
        angles = angles.repeat_interleave(2, dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.cos.to(t.dtype) + _rotate_half(t) * self.sin.to(t.dtype)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope: RoPE2D) -> torch.Tensor:
        b, n, d = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = rope(self.q_norm(heads(self.q(x)))), rope(self.k_norm(heads(self.k(x))))
        out = F.scaled_dot_product_attention(q, k, heads(self.v(x)))
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(2 / 3 * dim * mlp_ratio)
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(dim, hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class TrunkBlock(nn.Module):
    """Pre-norm block; conditioning enters through in-context tokens, not adaLN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = SwiGLU(dim, mlp_ratio)

    def forward(self, x: torch.Tensor, rope: RoPE2D) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope)
        return x + self.mlp(self.norm2(x))


class HeadBlock(TrunkBlock):
    """Wide DDT-head block, modulated per token by the trunk's output (adaLN-Zero)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__(dim, num_heads, mlp_ratio)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor, rope: RoPE2D) -> torch.Tensor:  # type: ignore[override]
        sh1, sc1, g1, sh2, sc2, g2 = self.adaln_modulation(c).chunk(6, dim=-1)
        x = x + g1 * self.attn(modulate(self.norm1(x), sh1, sc1), rope)
        return x + g2 * self.mlp(modulate(self.norm2(x), sh2, sc2))


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.linear = nn.Linear(dim, out_channels)
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaln_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))


class FourierTimeTokens(nn.Module):
    """Gaussian-Fourier timestep embedding broadcast onto `n_tokens` learnable tokens."""

    def __init__(self, dim: int, n_tokens: int = 4, fourier_dim: int = 256, scale: float = 1.0):
        super().__init__()
        self.register_buffer("W", torch.randn(fourier_dim) * scale)
        self.mlp = nn.Sequential(nn.Linear(2 * fourier_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.tokens = nn.Parameter(torch.randn(n_tokens, dim) / dim**0.5)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (base embedding (B,1,D), time tokens (B,n,D))."""
        ang = t[:, None].float() * self.W[None] * 2 * math.pi
        emb = self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1)).unsqueeze(1)
        return emb, self.tokens + emb


class AttributeTokens(nn.Module):
    """One token per face attribute; state 0 = unspecified (null), 1 = absent, 2 = present.

    Unspecified attributes use the same null embedding at train and test time, so any subset of
    attributes can be requested and all-null gives the unconditional model for CFG.
    """

    NULL, NO, YES = 0, 1, 2

    def __init__(self, dim: int, num_attributes: int, n_extra_tokens: int = 4):
        super().__init__()
        self.num_attributes = num_attributes
        self.state = nn.Embedding(num_attributes * 3, dim)
        self.slot = nn.Parameter(torch.randn(num_attributes, dim) / dim**0.5)
        self.extra = nn.Parameter(torch.randn(n_extra_tokens, dim) / dim**0.5)
        nn.init.normal_(self.state.weight, std=0.02)

    @property
    def n_tokens(self) -> int:
        return self.num_attributes + self.extra.shape[0]

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """attrs: (B, K) long in {0,1,2} -> (B, K + extra, D)."""
        idx = attrs + 3 * torch.arange(self.num_attributes, device=attrs.device)
        tok = self.slot + self.state(idx)
        return torch.cat([tok, self.extra.expand(attrs.shape[0], -1, -1)], dim=1)
