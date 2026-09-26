"""Measure the demo watermark: recovery under common edits, invisibility, and false positives.

    uv run --extra demo python scripts/watermark_eval.py --grid runs/eval/teacher_grid.png --data data/ffhq-256

Writes runs/eval/watermark.json.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from watermark import add_watermark, check_watermark

from humangen.data.parquet_images import decode_image, iter_rows, list_shards


def _roundtrip(im: Image.Image, fmt: str, **kw) -> Image.Image:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


ATTACKS = {
    "png": lambda im: _roundtrip(im, "PNG"),
    "jpeg_q75": lambda im: _roundtrip(im, "JPEG", quality=75),
    "jpeg_q50": lambda im: _roundtrip(im, "JPEG", quality=50),
    "downscale_2x_upscale": lambda im: im.resize((im.width // 2, im.height // 2), Image.BICUBIC).resize(im.size),
    "center_crop_90pct": lambda im: im.crop((im.width // 20, im.height // 20, im.width * 19 // 20, im.height * 19 // 20)),
}


def grid_tiles(path: str, n: int, tile: int = 256, pad: int = 2) -> list[Image.Image]:
    g = Image.open(path).convert("RGB")
    cols = (g.width - pad) // (tile + pad)
    out = []
    for i in range(n):
        x, y = pad + (i % cols) * (tile + pad), pad + (i // cols) * (tile + pad)
        if y + tile > g.height:
            break
        out.append(g.crop((x, y, x + tile, y + tile)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="runs/eval/teacher_grid.png", help="grid of generated faces (evaluate.py)")
    ap.add_argument("--data", default="data/ffhq-256", help="real faces for the false-positive check")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="runs/eval/watermark.json")
    a = ap.parse_args()
    gen = grid_tiles(a.grid, a.n)
    real = []
    for batch in iter_rows(list_shards(a.data), ["image"], batch_rows=256):
        real += [Image.fromarray(decode_image(c, 256)) for c in batch["image"]]
        if len(real) >= a.n:
            break
    real = real[: a.n]

    hits = dict.fromkeys(ATTACKS, 0)
    psnr, t0 = [], time.perf_counter()
    for im in gen:
        w = add_watermark(im)
        diff = np.asarray(w, np.float64) - np.asarray(im, np.float64)
        psnr.append(10 * np.log10(255**2 / max(float(np.mean(diff**2)), 1e-12)))
        for name, attack in ATTACKS.items():
            hits[name] += check_watermark(attack(w))
    embed_check_s = (time.perf_counter() - t0) / len(gen)
    fp_gen = sum(check_watermark(im) for im in gen)
    fp_real = sum(check_watermark(im) for im in real)
    result = {
        "method": "TrustMark Q (BCH text payload)",
        "n_marked": len(gen),
        "recovery": {k: round(v / len(gen), 4) for k, v in hits.items()},
        "psnr_db_mean": round(float(np.mean(psnr)), 2),
        "psnr_db_min": round(float(np.min(psnr)), 2),
        "false_positive_rate_unmarked_generated": round(fp_gen / len(gen), 4),
        "false_positive_rate_real_ffhq": round(fp_real / len(real), 4),
        "n_real": len(real),
        "cpu_seconds_per_image_embed_plus_5_checks": round(embed_check_s, 2),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
