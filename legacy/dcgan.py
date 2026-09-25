"""The original Human GAN (DCGAN, 64x64) with only its bugs fixed, used as the "before" baseline.

Architecture and hyper-parameters are unchanged from `dcgan_original.py`. Fixes:
  * Sigmoid + BCELoss -> logits + BCEWithLogitsLoss (numerically stable).
  * No silent all-black replacement for unreadable images (data comes from verified parquet shards).
  * `if __name__ == "__main__"` guard, fixed seed, checkpointing.
The dropped "HLC-GAN" latent-consistency loss is intentionally *not* reproduced: it penalises output
differences between nearby latents, which pushes the generator towards mode collapse.

    uv run python legacy/dcgan.py train --data data/ffhq-256 --minutes 40
    uv run python legacy/dcgan.py sample --n 5000 --out outputs/legacy_samples
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.utils import save_image

from humangen.data.parquet_images import decode_image, iter_rows, list_shards


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, ngf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # logits (the original applied Sigmoid here)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1)


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


def load_ffhq64(data: str, limit: int | None) -> torch.Tensor:
    imgs = []
    for batch in iter_rows(list_shards(data), ["image"], batch_rows=256):
        imgs.extend(decode_image(c, 64) for c in batch["image"])
        if limit and len(imgs) >= limit:
            break
    x = torch.from_numpy(np.stack(imgs[:limit] if limit else imgs)).permute(0, 3, 1, 2).float() / 127.5 - 1
    return x


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_ffhq64(args.data, args.limit)
    print(f"[dcgan] {len(data)} FFHQ images at 64x64", flush=True)
    g, d = Generator().to(dev), Discriminator().to(dev)
    g.apply(weights_init)
    d.apply(weights_init)
    opt_g = torch.optim.Adam(g.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(d.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = torch.randn(25, 100, 1, 1, device=dev)
    t0, step, epoch = time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < args.minutes * 60:
        perm = torch.randperm(len(data))
        for i in range(0, len(data) - 63, 64):
            real = data[perm[i : i + 64]].to(dev)
            real = torch.where(torch.rand(len(real), 1, 1, 1, device=dev) < 0.5, real, real.flip(-1))
            z = torch.randn(len(real), 100, 1, 1, device=dev)
            fake = g(z)
            ld = bce(d(real), torch.ones(len(real), device=dev)) + bce(d(fake.detach()), torch.zeros(len(real), device=dev))
            opt_d.zero_grad()
            ld.backward()
            opt_d.step()
            lg = bce(d(fake), torch.ones(len(real), device=dev))
            opt_g.zero_grad()
            lg.backward()
            opt_g.step()
            step += 1
        epoch += 1
        print(f"[dcgan] epoch {epoch} step {step} loss_d {ld.item():.3f} loss_g {lg.item():.3f} "
              f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    g.eval()
    with torch.no_grad():
        save_image(g(fixed) * 0.5 + 0.5, out / "samples.png", nrow=5)
    torch.save(g.state_dict(), out / "generator.pt")
    print(f"[dcgan] done: {epoch} epochs, {step} steps -> {out}", flush=True)


def sample(args: argparse.Namespace) -> None:
    from PIL import Image

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = Generator().to(dev)
    g.load_state_dict(torch.load(Path(args.ckpt), map_location=dev, weights_only=True))
    g.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    k = 0
    with torch.no_grad():
        while k < args.n:
            x = g(torch.randn(min(250, args.n - k), 100, 1, 1, device=dev)) * 0.5 + 0.5
            for im in (x.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8):
                Image.fromarray(im).save(out / f"{k:06d}.png")
                k += 1


def fid(args: argparse.Namespace) -> None:
    """Same protocol as scripts/evaluate.py: clean-fid vs the FFHQ-256 trainval70k reference, clean + legacy-TF."""
    import json
    import os
    import subprocess

    from cleanfid import fid as cfid

    from humangen.eval.metrics import frechet_distance

    cfid.frechet_distance = frechet_distance  # SciPy 1.18 removed sqrtm(disp=); see scripts/evaluate.py
    import shutil

    shutil.rmtree(args.out, ignore_errors=True)  # never score stale images from an earlier run
    sample(args)
    result = {"tag": "legacy_dcgan_64px", "ckpt": args.ckpt, "n": args.n, "resolution": 64,
              "note": "64x64 samples; clean-fid resizes them to the Inception input like every other model",
              "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    for mode in ("clean", "legacy_tensorflow"):
        result[f"fid_{mode}"] = round(cfid.compute_fid(
            args.out, dataset_name="ffhq", dataset_res=256, dataset_split="trainval70k", mode=mode,
            num_workers=0 if os.name == "nt" else 8, verbose=False), 3)
        print(f"[dcgan] FID ({mode}) = {result[f'fid_{mode}']}", flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data/ffhq-256")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--minutes", type=float, default=40)
    t.add_argument("--out", default="runs/legacy_dcgan")
    for name in ("sample", "fid"):
        s = sub.add_parser(name)
        s.add_argument("--ckpt", default="runs/legacy_dcgan/generator.pt")
        s.add_argument("--n", type=int, default=50000 if name == "fid" else 5000)
        s.add_argument("--out", default="outputs/legacy_samples")
        if name == "fid":
            s.add_argument("--json", default="runs/eval/legacy_dcgan_64px.json")
    args = ap.parse_args()
    {"train": train, "sample": sample, "fid": fid}[args.cmd](args)


if __name__ == "__main__":
    main()
