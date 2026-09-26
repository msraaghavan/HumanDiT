"""Measure the effect of the (fixed) autocast leak in scripts/evaluate.py: bf16 decode + DINOv2 features vs fp32.

Before the fix, generated images were decoded and featurised under bf16 autocast on Ampere+ GPUs, while real
FFHQ features were fp32. Same latents, both paths, against the pod's real FFHQ DINOv2 statistics (all 70k):

    uv run --extra eval python scripts/autocast_leak_check.py 2000
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from humangen.eval.generate import iterate_latents, load_generator, one_step_sampler
from humangen.eval.metrics import DINOv2Features, frechet_distance, knn_precision_recall, stats
from humangen.rae import RAE

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
model, _, _ = load_generator("runs/student_wflow/student.safetensors", dev)
lat = torch.cat(
    [z.cpu() for z, _ in iterate_latents(one_step_sampler(model, 1), N, 50, model.cfg.num_attributes, dev)]
)
del model
torch.cuda.empty_cache()
rae, dino = RAE(load_encoder=False).to(dev), DINOv2Features().to(dev)
real = np.load("runs/eval/ffhq256_dinov2_real.npz")
out, imgs_q = {}, {}
for mode in ("fp32", "bf16"):
    feats, qs = [], []
    for i in range(0, N, 25):
        z = lat[i : i + 25].to(dev)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if mode == "bf16" else torch.no_grad()
        with torch.no_grad(), ctx:
            im = rae.decode(z)
            feats.append(dino(im).float().cpu())
        qs.append((im.float().clamp(0, 1) * 255).round().byte().cpu())
    f = torch.cat(feats).numpy().astype(np.float64)
    imgs_q[mode] = torch.cat(qs)
    pr = knn_precision_recall(
        torch.from_numpy(real["subset"][:N]).to(dev), torch.from_numpy(f).float().to(dev)
    )
    out[mode] = {"fd_dinov2": frechet_distance(*stats(f), real["mu"], real["sigma"]), **pr}
    out[mode + "_feats"] = f
a, b = imgs_q["fp32"].float(), imgs_q["bf16"].float()
mse = (a - b).pow(2).mean().item()
res = {
    "n": N,
    "fp32": out["fp32"],
    "bf16_leak": out["bf16"],
    "fd_between_fp32_and_bf16_features": frechet_distance(
        *stats(out["fp32_feats"]), *stats(out["bf16_feats"])
    ),
    "image_psnr_bf16_vs_fp32_db": 10 * np.log10(255**2 / max(mse, 1e-12)),
    "image_max_abs_diff_levels": (a - b).abs().max().item(),
}
print(json.dumps(res, indent=2))
Path("runs/eval/autocast_leak_check.json").write_text(json.dumps(res, indent=2))
