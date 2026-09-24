"""Phase-1 gate: how well does the released RAEv2 decoder reconstruct FFHQ faces?

Reports PSNR, LPIPS (AlexNet) and optionally rFID (clean-fid, recon vs. originals), and saves an
originals/reconstructions grid. Encoder and decoder are loaded one at a time to fit in 4 GB.

    uv run python scripts/rae_reconstruction_gate.py --data data/ffhq-256 --n 500 --out outputs/rae_gate
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid, save_image

from humangen.data.parquet_images import decode_image, iter_rows, list_shards
from humangen.rae.autoencoder import (
    DECODER_DIR,
    DEFAULT_ROOT,
    ENCODER_WEIGHTS,
    LatentNormalizer,
    grid_to_tokens,
    tokens_to_grid,
)
from humangen.rae.decoder import RAEDecoder
from humangen.rae.encoder import DINOv3MultiLayerEncoder
from humangen.utils.device import get_precision


def load_images(data: str, n: int) -> torch.Tensor:
    imgs = []
    for batch in iter_rows(list_shards(data), ["image"], batch_rows=128):
        imgs.extend(decode_image(c, 256) for c in batch["image"])
        if len(imgs) >= n:
            break
    x = torch.from_numpy(np.stack(imgs[:n])).permute(0, 3, 1, 2).float() / 255.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ffhq-256")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="outputs/rae_gate")
    ap.add_argument("--rfid", action="store_true", help="also compute rFID with clean-fid (use n >= 10000)")
    args = ap.parse_args()

    prec = get_precision("fp32")  # measure the decoder faithfully, no reduced precision
    dev = prec.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)

    x = load_images(args.data, args.n)
    print(f"loaded {len(x)} images", flush=True)

    t0 = time.perf_counter()
    enc = DINOv3MultiLayerEncoder(str(root / ENCODER_WEIGHTS)).to(dev)
    norm = LatentNormalizer(root / DECODER_DIR / "stats.pt").to(dev)
    zs = []
    for i in range(0, len(x), args.batch):
        zs.append(norm.normalize(tokens_to_grid(enc(x[i : i + args.batch].to(dev)))).cpu())
    z = torch.cat(zs)
    t_enc = time.perf_counter() - t0
    del enc
    torch.cuda.empty_cache()
    print(f"encoded -> {tuple(z.shape)} in {t_enc:.1f}s; latent mean {z.mean():.3f} std {z.std():.3f}", flush=True)

    t0 = time.perf_counter()
    dec = RAEDecoder()
    dec.load_released(str(root / DECODER_DIR / "decoder.pt"))
    dec = dec.to(dev).eval()
    recs = []
    with torch.no_grad():
        for i in range(0, len(z), args.batch):
            zb = norm.denormalize(z[i : i + args.batch].to(dev))
            recs.append(dec(grid_to_tokens(zb)).clamp(0, 1).cpu())
    rec = torch.cat(recs)
    t_dec = time.perf_counter() - t0
    del dec
    torch.cuda.empty_cache()

    mse = ((rec - x) ** 2).flatten(1).mean(1)
    psnr = (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).mean().item()

    import lpips

    lp_fn = lpips.LPIPS(net="alex", verbose=False).to(dev)
    lp = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            lp.append(lp_fn(x[i : i + 32].to(dev) * 2 - 1, rec[i : i + 32].to(dev) * 2 - 1).flatten().cpu())
    lpips_v = torch.cat(lp).mean().item()

    metrics = {
        "n": len(x),
        "psnr_db": round(psnr, 3),
        "lpips_alex": round(lpips_v, 4),
        "encode_img_per_s": round(len(x) / t_enc, 2),
        "decode_img_per_s": round(len(x) / t_dec, 2),
        "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
    }

    if args.rfid:
        from cleanfid import fid

        for name, t in (("orig", x), ("rec", rec)):
            d = out / name
            d.mkdir(exist_ok=True)
            for j, im in enumerate(t):
                Image.fromarray((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(d / f"{j:06d}.png")
        metrics["rfid_clean"] = round(fid.compute_fid(str(out / "orig"), str(out / "rec"), mode="clean"), 4)

    k = min(16, len(x))
    grid = make_grid(torch.cat([x[:k], rec[:k]]), nrow=k, padding=2)
    save_image(grid, out / "recon_grid.png")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    assert math.isfinite(psnr)


if __name__ == "__main__":
    main()
