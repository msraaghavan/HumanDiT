"""Encode parquet image datasets into cached RAE latents (resumable numpy memmaps + manifest).

Layout under <out>/<name>/:
    latents.npy   float16 (N, F, 1024, 16, 16)  normalised RAE latents; F=2 stores [image, h-flip]
    pooled.npy    float32 (N, F, 1024)          spatial mean of the latents (attribute probes, audits)
    labels.npy    bool    (N, L)                optional boolean label columns (e.g. CelebA attributes)
    manifest.json shapes, sources, label names, progress (for resume)
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from humangen.data.parquet_images import decode_image


@dataclass
class CacheSpec:
    name: str
    shards: list[Path]
    size: int = 256
    flips: bool = True
    store_latents: bool = True
    label_columns: list[str] = field(default_factory=list)
    center_crop: bool = False  # CelebA: 178x218 -> 178x178 before resizing


class _ShardRows(IterableDataset):
    """Yields (global_index, uint8 HWC image, labels) with shards split across workers."""

    def __init__(self, spec: CacheSpec, offsets: list[int], start: int, limit: int):
        self.spec, self.offsets, self.start, self.limit = spec, offsets, start, limit

    def __iter__(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        cols = ["image", *self.spec.label_columns]
        for si, shard in enumerate(self.spec.shards):
            if si % nw != wid or self.offsets[si] >= self.limit:
                continue
            idx = self.offsets[si]
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=64, columns=cols):
                d = batch.to_pydict()
                for j, cell in enumerate(d["image"]):
                    if self.start <= idx < self.limit:
                        img = decode_image(cell)
                        if self.spec.center_crop:
                            h, w = img.shape[:2]
                            s = min(h, w)
                            top, left = (h - s) // 2, (w - s) // 2
                            img = img[top : top + s, left : left + s]
                        if img.shape[0] != self.spec.size:
                            from PIL import Image

                            img = np.asarray(
                                Image.fromarray(img).resize((self.spec.size,) * 2, Image.LANCZOS)
                            )
                        labels = np.array([bool(d[c][j]) for c in self.spec.label_columns], dtype=bool)
                        yield idx, img, labels
                    idx += 1
                    if idx >= self.limit:
                        break
                if idx >= self.limit:
                    break


def _collate(items):
    idx = torch.tensor([i for i, _, _ in items])
    imgs = torch.from_numpy(np.stack([im for _, im, _ in items]))
    labels = torch.from_numpy(np.stack([lb for _, _, lb in items]))
    return idx, imgs, labels


@torch.no_grad()
def build_cache(
    spec: CacheSpec,
    rae,  # humangen.rae.RAE with an encoder
    out_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    limit: int | None = None,
    autocast_dtype: torch.dtype | None = None,
    log_every: float = 30.0,
) -> Path:
    out = Path(out_root) / spec.name
    out.mkdir(parents=True, exist_ok=True)
    counts = [pq.ParquetFile(s).metadata.num_rows for s in spec.shards]
    offsets = list(np.cumsum([0, *counts[:-1]]))
    n = min(sum(counts), limit) if limit else sum(counts)
    nf = 2 if spec.flips else 1
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    fresh = manifest.get("n") != n or manifest.get("flips") != nf
    mode = "w+" if fresh else "r+"

    lat = (
        np.lib.format.open_memmap(out / "latents.npy", mode, np.float16, (n, nf, 1024, 16, 16))
        if spec.store_latents
        else None
    )
    pooled = np.lib.format.open_memmap(out / "pooled.npy", mode, np.float32, (n, nf, 1024))
    labels = (
        np.lib.format.open_memmap(out / "labels.npy", mode, bool, (n, len(spec.label_columns)))
        if spec.label_columns
        else None
    )
    done = 0 if fresh else int(manifest.get("done", 0))
    manifest.update(
        name=spec.name,
        n=n,
        flips=nf,
        latent_shape=[1024, 16, 16],
        latent_dtype="float16",
        shards=[s.name for s in spec.shards],
        label_names=spec.label_columns,
        encoder="DINOv3-L/16 layers 11..23 step 2 (RAEv2 K=7), RAEv2 general stats",
        done=done,
    )
    if done >= n:
        print(f"[cache] {spec.name}: complete ({n} rows)")
        return out

    # Rows are yielded in order within a shard but shards interleave across workers, so resume is
    # tracked conservatively as the count of rows written (a set of indices would also work).
    written = np.zeros(n, dtype=bool)
    written[:done] = True
    dev = next(rae.encoder.model.parameters()).device
    dl = DataLoader(
        _ShardRows(spec, offsets, done, n),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
        # forking after the TPU runtime has started can deadlock its threads; spawn clean workers instead
        multiprocessing_context="spawn" if dev.type == "xla" and num_workers > 0 else None,
    )
    t0, last = time.perf_counter(), 0.0
    for idx, imgs, lbl in dl:
        x = imgs.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        views = [x, x.flip(-1)] if spec.flips else [x]
        zs = []
        for v in views:
            if autocast_dtype is not None:
                with torch.autocast(dev.type, dtype=autocast_dtype):
                    z = rae.encode(v)
            else:
                z = rae.encode(v)
            zs.append(z.float())
        z = torch.stack(zs, dim=1)  # (B, F, C, H, W)
        ii = idx.numpy()
        if lat is not None:
            lat[ii] = z.half().cpu().numpy()
        pooled[ii] = z.mean(dim=(-2, -1)).cpu().numpy()
        if labels is not None:
            labels[ii] = lbl.numpy()
        written[ii] = True
        now = time.perf_counter() - t0
        if now - last > log_every:
            last = now
            k = int(written.sum())
            prefix = int(np.argmin(written)) if not written.all() else n
            manifest["done"] = prefix
            man_path.write_text(json.dumps(manifest, indent=2))
            rate = (k - done) / now
            print(
                f"[cache] {spec.name}: {k}/{n} ({rate:.1f} img/s, eta {(n - k) / max(rate, 1e-6) / 60:.1f} min)",
                flush=True,
            )
    for arr in (lat, pooled, labels):
        if arr is not None:
            arr.flush()
    assert written.all(), "some rows were not written"
    manifest["done"] = n
    manifest["seconds"] = round(time.perf_counter() - t0, 1)
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"[cache] {spec.name}: wrote {n} rows in {manifest['seconds']}s -> {out}")
    return out
