"""Representation autoencoder: frozen DINOv3 encoder + pixel decoder + latent normalisation."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from humangen.rae.decoder import DecoderConfig, RAEDecoder
from humangen.rae.encoder import DINOv3MultiLayerEncoder

DEFAULT_ROOT = Path(os.environ.get("HUMANGEN_PRETRAINED", "pretrained/raev2"))
ENCODER_WEIGHTS = "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DECODER_DIR = "stage1/general/dinov3l-k7"
HF_REPO = "nyu-visionx/RAEv2-models"

LATENT_SHAPE = (1024, 16, 16)


def ensure_weights(root: Path = DEFAULT_ROOT) -> Path:
    """Download the released encoder/decoder/stats if they are not present."""
    from huggingface_hub import hf_hub_download

    for rel in (ENCODER_WEIGHTS, f"{DECODER_DIR}/decoder.pt", f"{DECODER_DIR}/stats.pt"):
        if not (root / rel).exists():
            hf_hub_download(HF_REPO, rel, local_dir=str(root))
    return root


class LatentNormalizer(nn.Module):
    """Per-position, per-channel standardisation with the released RAEv2 statistics."""

    def __init__(self, stats_path: str | Path, eps: float = 1e-5):
        super().__init__()
        stats = torch.load(stats_path, map_location="cpu", weights_only=True)
        self.register_buffer("mean", stats["mean"].float())
        self.register_buffer("std", (stats["var"].float() + eps).sqrt())

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.mean) / self.std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean


def tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    b, n, c = tokens.shape
    h = int(n**0.5)
    return tokens.transpose(1, 2).reshape(b, c, h, h)


def grid_to_tokens(z: torch.Tensor) -> torch.Tensor:
    b, c, h, w = z.shape
    return z.reshape(b, c, h * w).transpose(1, 2)


class RAE(nn.Module):
    """encode: images [0,1] (B,3,256,256) -> normalised latents (B,1024,16,16); decode: the inverse."""

    def __init__(self, root: str | Path = DEFAULT_ROOT, load_encoder: bool = True, load_decoder: bool = True):
        super().__init__()
        root = Path(root)
        self.normalizer = LatentNormalizer(root / DECODER_DIR / "stats.pt")
        self.encoder = DINOv3MultiLayerEncoder(str(root / ENCODER_WEIGHTS)) if load_encoder else None
        self.decoder = None
        if load_decoder:
            self.decoder = RAEDecoder(DecoderConfig())
            self.decoder.load_released(str(root / DECODER_DIR / "decoder.pt"))
            self.decoder.requires_grad_(False).eval()

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        assert self.encoder is not None, "RAE was built without an encoder"
        return self.normalizer.normalize(tokens_to_grid(self.encoder(images)))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        assert self.decoder is not None, "RAE was built without a decoder"
        tokens = grid_to_tokens(self.normalizer.denormalize(z.float()))
        return self.decoder(tokens.to(next(self.decoder.parameters()).dtype)).float().clamp(0, 1)
