"""Training dataset over cached RAE latents with random horizontal-flip views and attribute labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from humangen.models.layers import AttributeTokens


def load_attributes(cache_dir: str | Path, n: int) -> tuple[np.ndarray, list[str]]:
    """(N, K) int8 in {0: unknown, 1: absent, 2: present}; all-unknown if no labels were built yet."""
    path = Path(cache_dir) / "attributes.npy"
    meta = Path(cache_dir) / "attributes.json"
    if path.exists() and meta.exists():
        names = json.loads(meta.read_text())["names"]
        return np.load(path)[:n], names
    return np.zeros((n, 0), dtype=np.int8), []


class LatentDataset(Dataset):
    def __init__(self, cache_dir: str | Path, limit: int | None = None, in_memory: bool = False):
        self.dir = Path(cache_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        assert manifest["done"] >= manifest["n"], f"cache {cache_dir} is incomplete"
        self.n = min(manifest["n"], limit) if limit else manifest["n"]
        self.flips = manifest["flips"]
        self.attrs, self.attr_names = load_attributes(self.dir, self.n)
        self._lat = np.load(self.dir / "latents.npy", mmap_mode="r")[: self.n] if in_memory else None
        if in_memory:
            self._lat = np.ascontiguousarray(self._lat)

    def _latents(self) -> np.ndarray:
        if self._lat is None:  # opened lazily so each DataLoader worker gets its own memmap
            self._lat = np.load(self.dir / "latents.npy", mmap_mode="r")
        return self._lat

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        f = int(torch.randint(self.flips, ()).item())
        z = torch.from_numpy(np.array(self._latents()[i, f]))  # float16 (C, H, W)
        a = (
            torch.from_numpy(self.attrs[i].astype(np.int64))
            if self.attrs.shape[1]
            else torch.zeros(0, dtype=torch.long)
        )
        return z, a


def drop_attributes(attrs: torch.Tensor, p_all: float, p_each: float) -> torch.Tensor:
    """CFG-style dropout: whole condition -> null with p_all; each attribute -> null with p_each."""
    if attrs.numel() == 0:
        return attrs
    keep_each = torch.rand(attrs.shape, device=attrs.device) >= p_each
    keep_all = (torch.rand(attrs.shape[0], 1, device=attrs.device) >= p_all).expand_as(attrs)
    return torch.where(keep_each & keep_all, attrs, torch.full_like(attrs, AttributeTokens.NULL))
