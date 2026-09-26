"""Distribution metrics: Fréchet distance, k-NN precision/recall, and DINOv2 feature extraction.

FID uses clean-fid (Parmar et al.) against its precomputed FFHQ-256 trainval70k statistics.
FD_DINOv2 follows Stein et al. (NeurIPS 2023): Fréchet distance in DINOv2 ViT-L/14 CLS space.
Precision/recall follows Kynkäänniemi et al. (2019) with k = 3.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)  # SciPy >= 1.18: `disp` argument removed
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = feats.astype(np.float64)
    return f.mean(0), np.cov(f, rowvar=False)


@torch.no_grad()
def knn_precision_recall(
    real: torch.Tensor, fake: torch.Tensor, k: int = 3, chunk: int = 2048
) -> dict[str, float]:
    """Improved precision/recall: a sample is covered if it lies inside some k-NN ball of the other set."""

    def radii(x: torch.Tensor) -> torch.Tensor:
        r = []
        for i in range(0, len(x), chunk):
            d = torch.cdist(x[i : i + chunk], x)
            r.append(d.kthvalue(k + 1, dim=1).values)  # +1 skips the point itself
        return torch.cat(r)

    def coverage(query: torch.Tensor, ref: torch.Tensor, ref_r: torch.Tensor) -> float:
        hit = []
        for i in range(0, len(query), chunk):
            d = torch.cdist(query[i : i + chunk], ref)
            hit.append((d <= ref_r[None]).any(dim=1))
        return torch.cat(hit).float().mean().item()

    real, fake = real.float(), fake.float()
    rr, rf = radii(real), radii(fake)
    return {"precision": coverage(fake, real, rr), "recall": coverage(real, fake, rf)}


class DINOv2Features(torch.nn.Module):
    """DINOv2 ViT-L/14 CLS features at 224 px, as used by FD_DINOv2 (dgm-eval)."""

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitl14", trust_repo=True, skip_validation=True
        )
        self.model.eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, images01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            images01.float(), size=(224, 224), mode="bicubic", antialias=True, align_corners=False
        )
        return self.model((x.clamp(0, 1) - self.mean) / self.std)


def cleanfid():
    """clean-fid 0.1.35 made to work here; returns (cleanfid.fid module, num_workers).

    * SciPy 1.18 removed `sqrtm(disp=)`, which clean-fid calls: use `frechet_distance` above instead.
    * Its dataset holds a local resize function that spawn/forkserver workers cannot pickle. Forkserver is the
      default on Linux since Python 3.14, so use fork there (the workers only decode and resize images and
      never touch CUDA); Windows has no fork, so load in the main process.
    """
    from cleanfid import fid

    fid.frechet_distance = frechet_distance
    if os.name == "nt":
        return fid, 0
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    return fid, 8
