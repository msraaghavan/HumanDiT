# Feature extraction follows RAEv2's `DINOv3MultiLayerSimpleAddEncoder`
# (https://github.com/nanovisionx/RAEv2, CC BY-NC 4.0) exactly, so the released latent
# statistics and decoders apply unchanged.
"""Frozen DINOv3 encoder producing RAEv2 multi-layer latents."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# Pinned upstream commit, identical to the one RAEv2 uses.
DINOV3_HUB_REF = "facebookresearch/dinov3:94a96ac83c2446f15f9bdcfae23cad3c6a9d4988"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# RAEv2 "K=7": every other block of the last half of DINOv3-L (24 blocks).
DINOV3L_K7_LAYERS = (11, 13, 15, 17, 19, 21, 23)


def load_dinov3(arch: str, weights: str) -> nn.Module:
    return torch.hub.load(
        DINOV3_HUB_REF, arch, source="github", trust_repo=True, skip_validation=True, weights=weights
    )


class DINOv3MultiLayerEncoder(nn.Module):
    """Images in [0, 1] -> patch tokens (B, (H/16)*(W/16), D).

    Latent = mean over the selected blocks of the (non-affine) layer-normed patch tokens,
    plus the mean patch token of the last selected block broadcast to every position.
    """

    def __init__(
        self,
        weights: str,
        arch: str = "dinov3_vitl16",
        layers: tuple[int, ...] = DINOV3L_K7_LAYERS,
        resolution: int = 256,
    ):
        super().__init__()
        self.model = load_dinov3(arch, weights)
        self.embed_dim: int = self.model.embed_dim
        # RAEv2 strips the final LayerNorm's affine parameters.
        self.model.norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)
        self.layers = list(layers)
        self.resolution = resolution
        self.patch_size = 16
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> DINOv3MultiLayerEncoder:
        return super().train(False)  # always frozen

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Match RAEv2: x*255 -> /255 -> antialiased resize -> ImageNet normalisation.
        x = images.float()
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            x = F.interpolate(x, size=(self.resolution,) * 2, mode="bilinear", antialias=True)
        x = (x - self.mean) / self.std
        outs = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=False, return_class_token=False, norm=True
        )
        tokens = torch.stack(outs, dim=0).mean(dim=0)
        return tokens + outs[-1].mean(dim=1, keepdim=True)
