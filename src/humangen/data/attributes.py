"""DINOv3 linear-probe face-attribute labeller (trained on CelebA, applied to FFHQ and to generated latents).

Features are spatially mean-pooled RAE latents, so the same probe scores real images and generator
outputs without decoding. Labels abstain when uncertain: thresholds are calibrated on the CelebA
validation split so that "present" has >= `min_precision` precision and "absent" has >= `min_npv`
negative predictive value; anything in between is left unspecified (null) for training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Excluded: subjective/appearance-judgement labels, image-quality labels, and skin tone.
EXCLUDED = {
    "Attractive", "Blurry", "Big_Lips", "Big_Nose", "Pointy_Nose", "Oval_Face", "Narrow_Eyes",
    "Arched_Eyebrows", "Bags_Under_Eyes", "High_Cheekbones", "Rosy_Cheeks", "Chubby", "Double_Chin",
    "Pale_Skin",
}  # fmt: skip
# Friendlier names; "No_Beard" is inverted into "Beard".
RENAME = {
    "No_Beard": "Beard",
    "Mouth_Slightly_Open": "Mouth_Open",
    "Wearing_Hat": "Hat",
    "Wearing_Lipstick": "Lipstick",
}
INVERT = {"No_Beard"}


@dataclass
class Probe:
    names: list[str]  # CelebA column names (before RENAME)
    weight: torch.Tensor  # (K, D)
    bias: torch.Tensor  # (K,)
    mu: torch.Tensor  # (D,) feature standardisation
    sigma: torch.Tensor  # (D,)
    lo: np.ndarray  # (K,) prob <= lo -> absent
    hi: np.ndarray  # (K,) prob >= hi -> present

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        x = (feats.float() - self.mu.to(feats.device)) / self.sigma.to(feats.device)
        return x @ self.weight.to(feats.device).T + self.bias.to(feats.device)

    def probs(self, feats: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.logits(feats))
        inv = torch.tensor([n in INVERT for n in self.names], device=p.device)
        return torch.where(inv, 1 - p, p)  # probability that the *display* attribute is present

    @property
    def display_names(self) -> list[str]:
        return [RENAME.get(n, n) for n in self.names]

    def ternary(self, feats: torch.Tensor) -> np.ndarray:
        """0 = unknown, 1 = absent, 2 = present (display semantics, i.e. after inversion)."""
        p = torch.sigmoid(self.logits(feats)).cpu().numpy()
        out = np.zeros(p.shape, dtype=np.int8)
        out[p >= self.hi] = 2
        out[p <= self.lo] = 1
        inv = np.array([n in INVERT for n in self.names])
        swapped = out.copy()
        swapped[:, inv] = np.where(out[:, inv] == 2, 1, np.where(out[:, inv] == 1, 2, 0))
        return swapped

    def save(self, path: str | Path) -> None:
        torch.save(
            {k: getattr(self, k) for k in ("names", "weight", "bias", "mu", "sigma", "lo", "hi")}, path
        )

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        return cls(**torch.load(path, map_location="cpu", weights_only=False))


def fit_linear_probe(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    epochs: int = 30,
    min_steps: int = 3000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Multi-label logistic regression with per-attribute class balancing; weight decay picked on val."""
    from sklearn.metrics import balanced_accuracy_score  # not needed by the demo

    xt = torch.from_numpy(x).float().to(device)
    mu, sigma = xt.mean(0), xt.std(0).clamp_min(1e-6)
    xt = (xt - mu) / sigma
    yt = torch.from_numpy(y).float().to(device)
    xv = (torch.from_numpy(x_val).float().to(device) - mu) / sigma
    pos = yt.mean(0).clamp(1e-3, 1 - 1e-3)
    pos_weight = (1 - pos) / pos  # balance positives and negatives per attribute
    batch = 1024
    epochs = max(epochs, -(-min_steps // -(-len(xt) // batch)))  # small datasets still get min_steps
    best = None
    for wd in (1e-4, 1e-3, 1e-2):
        torch.manual_seed(0)
        lin = torch.nn.Linear(xt.shape[1], yt.shape[1]).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=wd)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        for _ in range(epochs):
            perm = torch.randperm(len(xt), device=device)
            for i in range(0, len(xt), batch):
                idx = perm[i : i + batch]
                opt.zero_grad()
                loss_fn(lin(xt[idx]), yt[idx]).backward()
                opt.step()
        with torch.no_grad():
            pv = (lin(xv) > 0).cpu().numpy()
        score = np.mean([balanced_accuracy_score(y_val[:, k], pv[:, k]) for k in range(y.shape[1])])
        if best is None or score > best[0]:
            best = (score, lin.weight.detach().cpu(), lin.bias.detach().cpu())
    _, w, b = best
    return w, b, mu.cpu(), sigma.cpu()


def calibrate_abstention(
    p_val: np.ndarray, y_val: np.ndarray, min_precision: float, min_npv: float, min_count: int = 20
):
    """Per attribute: hi = lowest prob threshold with precision >= min_precision (present),
    lo = highest threshold with NPV >= min_npv (absent).

    Every validation probability is a candidate threshold. The probe is trained class-balanced, so for rare
    attributes (Bald, Mustache, ...) 90% precision is only reached above p = 0.99, which a fixed 0.01-0.99 grid
    never tries (it left those attributes with no "present" labels at all).
    """
    k = p_val.shape[1]
    lo, hi = (
        np.full(k, -1e-6),
        np.full(k, 1 + 1e-6),
    )  # unreachable unless calibrated (float32 sigmoid can hit 1.0)
    for j in range(k):
        p, y = p_val[:, j], y_val[:, j].astype(bool)
        order = np.argsort(-p, kind="stable")  # present: take the top-n scores, n as large as possible
        n = np.arange(1, len(p) + 1)
        ok = (n >= min_count) & (np.cumsum(y[order]) / n >= min_precision)
        if ok.any():
            hi[j] = p[order][np.flatnonzero(ok)[-1]]
        order = np.argsort(p, kind="stable")  # absent: take the bottom-n scores
        ok = (n >= min_count) & (np.cumsum(~y[order]) / n >= min_npv)
        if ok.any():
            lo[j] = p[order][np.flatnonzero(ok)[-1]]
        lo[j] = min(lo[j], hi[j])
    return lo, hi


def evaluate(probe: Probe, x: np.ndarray, y: np.ndarray) -> dict[str, dict[str, float]]:
    from sklearn.metrics import (
        balanced_accuracy_score,
        roc_auc_score,
    )  # training/evaluation only: the demo never needs it

    p = torch.sigmoid(probe.logits(torch.from_numpy(x))).numpy()
    out = {}
    for j, name in enumerate(probe.names):
        yj = y[:, j]
        out[name] = {
            "balanced_acc": float(balanced_accuracy_score(yj, p[:, j] > 0.5)),
            "auroc": float(roc_auc_score(yj, p[:, j])) if 0 < yj.mean() < 1 else float("nan"),
            "positive_rate": float(yj.mean()),
        }
    return out


def select_attributes(metrics: dict[str, dict[str, float]], min_bacc: float, max_k: int) -> list[str]:
    ok = [
        (m["balanced_acc"], n)
        for n, m in metrics.items()
        if n not in EXCLUDED and m["balanced_acc"] >= min_bacc
    ]
    return [n for _, n in sorted(ok, reverse=True)[:max_k]]


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2))
