"""Cross-platform agreement + speed check for the accelerators the follow-up jobs could run on.

Runs the exact code paths of the real jobs on fixed inputs and saves their outputs, so different accelerators (and
precisions) can be compared against a full-precision reference:
    teacher   64 samples, IG 1.55, 50 steps, bf16 autocast -- as in evaluate.py (noise is drawn on the CPU, so it is
              bit-identical everywhere); decoded to uint8 images as for FID
    encoder   32 fixed FFHQ images through the RAEv2 encoder at full precision -- as in prepare_data.py
With --timing it also measures the workloads that dominate the real run: 50-step sampling at the eval batch size,
encoding, and a bf16 forward+backward training step.

    python scripts/platform_check.py --ckpt runs/teacher_b/ckpt/ema.safetensors --images fixed.npy --out X [--timing]
    python scripts/platform_check.py ... --fp32          # reference: no autocast; no TF32 / single-pass-bf16 matmuls
    python scripts/platform_check.py --compare REF_DIR OTHER_DIR [...]
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from humangen.diffusion.guidance import GuidanceConfig
from humangen.eval.generate import iterate_latents, load_generator, teacher_sampler
from humangen.rae.autoencoder import RAE, ensure_weights
from humangen.utils.device import full_precision_matmuls, get_precision, sync


def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "xla":
        import os

        return f"TPU {os.environ.get('TPU_ACCELERATOR_TYPE', '?')}"
    return platform.processor() or "cpu"


def timed(fn, dev: torch.device, warmup: int, reps: int) -> float:
    """Seconds per call after `warmup` calls (XLA compiles on the first call of each shape)."""
    for _ in range(warmup):
        fn()
    sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    sync(dev)
    return (time.perf_counter() - t) / reps


def run(args: argparse.Namespace) -> None:
    prec = get_precision("fp32" if args.fp32 else None)
    dev = prec.device
    if args.fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        full_precision_matmuls(dev)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ensure_weights()
    meta: dict = {
        "device": device_name(dev),
        "precision": prec.name,
        "torch": torch.__version__,
        "fp32_reference": args.fp32,
    }

    # teacher: same sampler, guidance and noise as evaluate.py (its first samples at seed 0)
    model, _, _ = load_generator(args.ckpt, dev)
    k = model.cfg.num_attributes
    sample_fn = teacher_sampler(
        model, GuidanceConfig(ig_scale=args.ig_scale, ig_t_min=0.10, cfg_scale=1.0), 50
    )
    rae = RAE().to(dev)
    z, _ = next(iterate_latents(sample_fn, args.n, args.n, k, dev, seed=0, autocast=prec.autocast))
    with torch.no_grad():
        imgs = torch.cat([rae.decode(z[i : i + 32]) for i in range(0, len(z), 32)])
    np.save(out / "teacher_z.npy", z.cpu().numpy())
    np.save(out / "teacher_img.npy", (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8))

    # encoder: always full precision, as in prepare_data.py
    full_precision_matmuls(dev)
    x = torch.from_numpy(np.load(args.images)).permute(0, 3, 1, 2).float().div(255).to(dev)
    enc = rae.encode(x)
    np.save(out / "encoded_z.npy", enc.float().cpu().numpy())

    if args.timing:
        b = args.eval_batch
        noise = torch.randn(b, 1024, 16, 16, generator=torch.Generator().manual_seed(1)).to(dev)
        attrs = torch.zeros(b, k, dtype=torch.long, device=dev)

        def sample() -> None:
            with torch.no_grad(), prec.autocast():
                sample_fn(noise, attrs)

        xb = x.repeat(2, 1, 1, 1)[:64]
        model.requires_grad_(True)
        xt = torch.randn(64, 1024, 16, 16, device=dev)
        tt = torch.rand(64, device=dev)
        at = torch.zeros(64, k, dtype=torch.long, device=dev)

        def train_step() -> None:
            with prec.autocast():
                loss = model(xt, tt, at)[0].float().pow(2).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            sync(dev)

        meta["sample_50step_img_per_s"] = round(b / timed(sample, dev, 1, 1), 2)
        meta["encode_img_per_s"] = round(64 / timed(lambda: rae.encode(xb), dev, 1, 3), 1)
        meta["train_step_b64_s"] = round(timed(train_step, dev, 3, 10), 4)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def psnr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).reshape(len(a), -1).mean(1)
    return 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12))


def rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (a - b).reshape(len(a), -1)
    return np.linalg.norm(d, axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1)


def compare(dirs: list[str]) -> None:
    ref = Path(dirs[0])
    r = {k: np.load(ref / f"{k}.npy") for k in ("teacher_z", "teacher_img", "encoded_z")}
    rm = json.loads((ref / "meta.json").read_text())
    print(f"reference: {ref.name} ({rm['device']}, {rm['precision']})")
    print(
        f"{'run':28s} {'teacher latent rel.err':>24s} {'teacher image PSNR dB':>24s} {'encoder rel.err':>18s}"
    )
    print(f"{'':28s} {'mean / max':>24s} {'mean / min':>24s} {'mean / max':>18s}")
    for d in map(Path, dirs[1:]):
        o = {k: np.load(d / f"{k}.npy") for k in r}
        m = json.loads((d / "meta.json").read_text())
        tz, ti = rel_err(o["teacher_z"], r["teacher_z"]), psnr(o["teacher_img"], r["teacher_img"])
        ez = rel_err(o["encoded_z"], r["encoded_z"])
        print(
            f"{d.name + ' (' + m['precision'] + ')':28s} {tz.mean():11.2e} / {tz.max():9.2e}  "
            f"{ti.mean():11.1f} / {ti.min():9.1f}  {ez.mean():8.2e} / {ez.max():8.2e}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", metavar="DIR", help="reference dir first")
    ap.add_argument("--ckpt")
    ap.add_argument("--images", help="uint8 (N,256,256,3) .npy of fixed real images")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--ig-scale", type=float, default=1.55)
    ap.add_argument("--eval-batch", type=int, default=125, help="evaluate.py's default batch")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--timing", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
