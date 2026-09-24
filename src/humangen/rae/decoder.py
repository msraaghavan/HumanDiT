# Adapted from RAEv2 (https://github.com/nanovisionx/RAEv2, src/stage1/decoders/decoder.py),
# Copyright (c) 2025-2026 Singh, Zheng, Wu, Zhang, Shechtman, Xie. Licensed CC BY-NC 4.0.
# Changes: compact rewrite without the transformers config/output classes, and attention via
# F.scaled_dot_product_attention. Parameter names are unchanged, so released checkpoints load strictly.
"""ViT-MAE-style pixel decoder that maps RAE latents (B, 256, 1024) to 256x256 RGB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DecoderConfig:
    latent_dim: int = 1024  # RAE latent channels (DINOv3-L width)
    hidden_size: int = 1152  # ViT-XL decoder width
    num_layers: int = 28
    num_heads: int = 16
    intermediate_size: int = 4096
    patch_size: int = 16
    num_patches: int = 256
    num_channels: int = 3
    layer_norm_eps: float = 1e-12

    @property
    def image_size(self) -> int:
        return self.patch_size * int(self.num_patches**0.5)


def _sincos_1d(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_pos_embed_2d(embed_dim: int, grid_size: int, add_cls_token: bool = True) -> torch.Tensor:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb = np.concatenate([_sincos_1d(embed_dim // 2, grid[0]), _sincos_1d(embed_dim // 2, grid[1])], axis=1)
    if add_cls_token:
        emb = np.concatenate([np.zeros([1, embed_dim]), emb], axis=0)
    return torch.from_numpy(emb).float().unsqueeze(0)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.num_heads, d // self.num_heads).transpose(1, 2)

        out = F.scaled_dot_product_attention(heads(self.query(x)), heads(self.key(x)), heads(self.value(x)))
        return out.transpose(1, 2).reshape(b, n, d)


class _Dense(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.dense = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attention = _SelfAttention(dim, num_heads)
        self.output = _Dense(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(x))


class _Layer(nn.Module):
    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        self.attention = _Attention(cfg.hidden_size, cfg.num_heads)
        self.intermediate = _Dense(cfg.hidden_size, cfg.intermediate_size)
        self.output = _Dense(cfg.intermediate_size, cfg.hidden_size)
        self.layernorm_before = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layernorm_before(x))
        return x + self.output(F.gelu(self.intermediate(self.layernorm_after(x))))


class RAEDecoder(nn.Module):
    """Decodes RAE latent tokens (B, N, C) into images (B, 3, H, W) in roughly [0, 1]."""

    def __init__(self, cfg: DecoderConfig | None = None):
        super().__init__()
        cfg = cfg or DecoderConfig()
        self.cfg = cfg
        self.decoder_embed = nn.Linear(cfg.latent_dim, cfg.hidden_size)
        self.register_buffer(
            "decoder_pos_embed",
            sincos_pos_embed_2d(cfg.hidden_size, int(cfg.num_patches**0.5)),
            persistent=True,
        )
        self.trainable_cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.decoder_layers = nn.ModuleList(_Layer(cfg) for _ in range(cfg.num_layers))
        self.decoder_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.decoder_pred = nn.Linear(cfg.hidden_size, cfg.patch_size**2 * cfg.num_channels)
        self.gradient_checkpointing = False

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p, c = self.cfg.patch_size, self.cfg.num_channels
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, c)
        return torch.einsum("nhwpqc->nchpwq", x).reshape(x.shape[0], c, h * p, w * p)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(tokens)
        x = torch.cat([self.trainable_cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.decoder_pos_embed.to(x.dtype)
        for layer in self.decoder_layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.decoder_pred(self.decoder_norm(x))[:, 1:]
        return self.unpatchify(x)

    def load_released(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
