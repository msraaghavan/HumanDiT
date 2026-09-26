"""Measure how guidance shifts latent statistics: generated vs real RAEv2 latents (std and pooled-vector norm).

    uv run python scripts/latent_stats.py --ckpt runs/teacher_b/ckpt/ema.safetensors --ig 1.0 1.78

Writes runs/eval/latent_stats.json. The README uses it to explain why probe-based metrics over-fire on samples
generated with strong guidance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humangen.diffusion.guidance import GuidanceConfig, guided_model_fn
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.eval.generate import load_generator
from humangen.utils.device import get_precision


def summary(z: torch.Tensor) -> dict:
    return {
        "std": round(float(z.std()), 4),
        "pooled_norm": round(float(z.mean(dim=(-2, -1)).norm(dim=1).mean()), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", default="cache/ffhq256")
    ap.add_argument("--ig", type=float, nargs="+", default=[1.0, 1.78])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="runs/eval/latent_stats.json")
    a = ap.parse_args()
    prec = get_precision()
    model, _, meta = load_generator(a.ckpt, prec.device)
    lat = np.load(Path(a.cache) / "latents.npy", mmap_mode="r")
    idx = np.random.default_rng(0).choice(len(lat), min(256, len(lat)), replace=False)
    result = {
        "ckpt": a.ckpt,
        "ckpt_step": meta.get("step"),
        "n": a.n,
        "steps": a.steps,
        "real": summary(torch.from_numpy(np.array(lat[np.sort(idx), 0])).float()),
    }
    transport = Transport(shift=dimension_shift(1024 * 16 * 16))
    for ig in a.ig:
        zs = []
        for i in range(0, a.n, a.batch):
            g = torch.Generator().manual_seed(i)
            noise = torch.randn(min(a.batch, a.n - i), 1024, 16, 16, generator=g).to(prec.device)
            attrs = torch.zeros(len(noise), model.cfg.num_attributes, dtype=torch.long, device=prec.device)
            with torch.no_grad(), prec.autocast():
                zs.append(
                    transport.sample(
                        guided_model_fn(model, attrs, GuidanceConfig(ig_scale=ig)), noise, a.steps
                    )
                    .float()
                    .cpu()
                )
        result[f"ig_{ig:.2f}"] = summary(torch.cat(zs))
        print(f"[stats] IG {ig:.2f}: {result[f'ig_{ig:.2f}']} (real {result['real']})", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
