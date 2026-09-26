"""Re-render the trainer's fixed-seed sample grid (Trainer.sample_grid) at any guidance scale: README hero image.

Same noise (CPU generator, seed 1234), same layout: row 1 unconditional, row 2 one attribute per face, CFG 2.0.

    uv run python scripts/hero_grid.py --ig-scale 1.55 --out docs/assets/teacher_final_grid.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from humangen.diffusion.guidance import GuidanceConfig
from humangen.eval.generate import load_generator, teacher_sampler
from humangen.models.layers import AttributeTokens
from humangen.rae import RAE
from humangen.utils.device import get_precision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--ig-scale", type=float, default=1.55)
    ap.add_argument("--cfg-scale", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="docs/assets/teacher_final_grid.jpg")
    a = ap.parse_args()

    prec = get_precision()
    model, names, _ = load_generator(a.ckpt, prec.device)
    n, k = 16, model.cfg.num_attributes
    noise = torch.randn(n, 1024, 16, 16, generator=torch.Generator(device="cpu").manual_seed(1234))
    attrs = torch.zeros(n, k, dtype=torch.long)
    for j in range(8, n):  # row 2: one attribute switched on per face, as in Trainer.sample_grid
        attrs[j, (j - 8) % k] = AttributeTokens.YES
    sample = teacher_sampler(model, GuidanceConfig(ig_scale=a.ig_scale, cfg_scale=a.cfg_scale), a.steps)
    zs = []
    with torch.no_grad(), prec.autocast():
        for i in range(0, n, 4):  # 4 at a time fits a 4 GB GPU
            zs.append(
                sample(noise[i : i + 4].to(prec.device), attrs[i : i + 4].to(prec.device)).float().cpu()
            )
    del model
    rae = RAE(load_encoder=False).to(prec.device)
    with torch.no_grad():
        imgs = torch.cat([rae.decode(z.to(prec.device)).float().cpu() for z in zs])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(make_grid(imgs.clamp(0, 1), nrow=8, padding=2)).save(out, quality=90)
    print(f"row 2 attributes: {[names[(j - 8) % k] for j in range(8, n)]}")
    print(f"saved {out} (IG {a.ig_scale}, CFG {a.cfg_scale}, {a.steps} steps)")


if __name__ == "__main__":
    main()
