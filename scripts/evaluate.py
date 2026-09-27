"""Evaluate a generator: FID (clean + legacy-TF), FD_DINOv2, precision/recall, controllability,
memorisation audit, and sampling speed. Every number in the README comes from these JSON files.

    python scripts/evaluate.py --ckpt runs/teacher_b/ckpt/ema.safetensors --n 50000 --steps 50 --tag teacher_50k
    python scripts/evaluate.py --ckpt runs/student_wflow/ckpt/student.safetensors --kind student --steps 1 --tag wflow_1step
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from humangen.data.attributes import Probe
from humangen.data.parquet_images import decode_image, iter_rows, list_shards
from humangen.diffusion.guidance import GuidanceConfig
from humangen.eval.generate import iterate_latents, load_generator, one_step_sampler, teacher_sampler
from humangen.eval.metrics import DINOv2Features, cleanfid, frechet_distance, knn_precision_recall, stats
from humangen.models.layers import AttributeTokens
from humangen.rae import RAE
from humangen.utils.device import get_precision


def real_dinov2_stats(
    feat: DINOv2Features, data: str, out: Path, dev, n_pr: int, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DINOv2 features of all real FFHQ images (cached): (mu, sigma, subset for precision/recall)."""
    if out.exists():
        z = np.load(out)
        return z["mu"], z["sigma"], z["subset"]
    feats, buf, seen = [], [], 0
    for batch in iter_rows(list_shards(data), ["image"], batch_rows=256):
        if limit is not None and seen >= limit:
            break
        cells = batch["image"][: None if limit is None else limit - seen]
        seen += len(cells)
        buf.extend(decode_image(c, 256) for c in cells)
        while len(buf) >= 128:
            x = torch.from_numpy(np.stack(buf[:128])).permute(0, 3, 1, 2).to(dev).float() / 255
            feats.append(feat(x).cpu().numpy())
            buf = buf[128:]
    if buf:
        x = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).to(dev).float() / 255
        feats.append(feat(x).cpu().numpy())
    f = np.concatenate(feats)
    mu, sigma = stats(f)
    subset = f[np.random.default_rng(0).permutation(len(f))[:n_pr]]
    np.savez(out, mu=mu, sigma=sigma, subset=subset, n=len(f))
    return mu, sigma, subset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--kind", choices=["teacher", "student"], default="teacher")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=125)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--method", default="euler")
    ap.add_argument("--ig-scale", type=float, default=1.78)
    ap.add_argument("--ig-tmin", type=float, default=0.10)
    ap.add_argument("--cfg-scale", type=float, default=2.0, help="attribute CFG for the controllability test")
    ap.add_argument("--metrics", default="fid,fd_dinov2,pr,control,memo,speed")
    ap.add_argument("--data", default="data/ffhq-256")
    ap.add_argument("--cache", default="cache/ffhq256")
    ap.add_argument("--probe", default="cache/attribute_probe.pt")
    ap.add_argument("--out", default="runs/eval")
    ap.add_argument("--keep-images", action="store_true")
    ap.add_argument("--control-n", type=int, default=256)
    ap.add_argument("--control-batch", type=int, default=64, help="lower it on small GPUs (4 GB: 16)")
    ap.add_argument(
        "--real-limit", type=int, default=None, help="real images for DINOv2 stats (default: all)"
    )
    args = ap.parse_args()
    metrics = set(args.metrics.split(","))

    prec = get_precision()
    dev = prec.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = out / f"{args.tag}_images"
    model, attr_names, meta = load_generator(args.ckpt, dev)
    k = model.cfg.num_attributes
    guidance = GuidanceConfig(ig_scale=args.ig_scale, ig_t_min=args.ig_tmin, cfg_scale=1.0)
    sample_fn = (
        teacher_sampler(model, guidance, args.steps, args.method)
        if args.kind == "teacher"
        else one_step_sampler(model, args.steps)
    )
    rae = RAE(load_encoder=False)  # kept on the CPU until a metric decodes (1.7 GB of fp32 weights)
    result: dict = {
        "tag": args.tag,
        "ckpt": args.ckpt,
        "kind": args.kind,
        "steps": args.steps,
        "n": args.n,
        "guidance": vars(guidance),
        "gpu": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
        "ckpt_step": meta.get("step"),
        "samples_seen": meta.get("samples_seen"),
    }
    try:
        result["git"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        result["git"] = None

    # ---------------------------------------------------------------- unconditional generation pass
    need_images = metrics & {"fid", "fd_dinov2", "pr"}
    dino = DINOv2Features().to(dev) if metrics & {"fd_dinov2", "pr"} else None
    pooled, dfeats, preview = [], [], []
    if need_images or "memo" in metrics:
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True)
        if need_images:
            rae.to(dev)
        pool = ThreadPoolExecutor(8)
        t0, written = time.perf_counter(), 0
        for z, _ in iterate_latents(sample_fn, args.n, args.batch, k, dev, seed=0, autocast=prec.autocast):
            pooled.append(z.mean(dim=(-2, -1)).cpu())
            if need_images:
                with torch.no_grad():
                    imgs = torch.cat([rae.decode(z[i : i + 32]) for i in range(0, len(z), 32)])
                if dino is not None:
                    dfeats.append(dino(imgs).cpu())
                if sum(len(p) for p in preview) < 64:
                    preview.append(imgs[:64].cpu())
                arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
                for j, a in enumerate(arr):
                    pool.submit(Image.fromarray(a).save, img_dir / f"{written + j:06d}.png")
                written += len(arr)
            if written and written % (args.batch * 40) == 0:
                print(
                    f"[eval] {written}/{args.n} images ({written / (time.perf_counter() - t0):.1f}/s)",
                    flush=True,
                )
        pool.shutdown(wait=True)
        result["generation_seconds"] = round(time.perf_counter() - t0, 1)
        if preview:
            save_image(torch.cat(preview)[:64], out / f"{args.tag}_grid.png", nrow=8, padding=2)

    # Free GPU memory between metric networks (the decoder alone is 415M params; matters on small GPUs).
    rae.cpu()
    torch.cuda.empty_cache()
    if dino is not None:
        mu_r, sig_r, sub_r = real_dinov2_stats(
            dino, args.data, out / "ffhq256_dinov2_real.npz", dev, 10000, args.real_limit
        )
        g = torch.cat(dfeats).numpy()
        if "fd_dinov2" in metrics:
            result["fd_dinov2"] = round(frechet_distance(*stats(g), mu_r, sig_r), 2)
            print(f"[eval] FD_DINOv2 = {result['fd_dinov2']}", flush=True)
        if "pr" in metrics:
            m = min(len(g), len(sub_r), 10000)
            pr_dev = dev if dev.type == "cuda" else torch.device("cpu")  # kthvalue/cdist: CPU on TPU hosts
            pr = knn_precision_recall(
                torch.from_numpy(sub_r[:m]).to(pr_dev), torch.from_numpy(g[:m]).to(pr_dev)
            )
            result.update({k2: round(v, 4) for k2, v in pr.items()})
            print(f"[eval] precision {pr['precision']:.3f} recall {pr['recall']:.3f}", flush=True)
        del dino
        torch.cuda.empty_cache()

    if "fid" in metrics:
        fid, workers = cleanfid()  # SciPy 1.18 + Python 3.14 workarounds; see humangen.eval.metrics.cleanfid
        for mode in ("clean", "legacy_tensorflow"):
            result[f"fid_{mode}"] = round(
                fid.compute_fid(
                    str(img_dir),
                    dataset_name="ffhq",
                    dataset_res=256,
                    dataset_split="trainval70k",
                    mode=mode,
                    num_workers=workers,
                    device=dev
                    if dev.type == "cuda"
                    else torch.device("cpu"),  # TPU: Inception on the host CPU
                    use_dataparallel=False,
                    verbose=False,
                ),
                3,
            )
            print(f"[eval] FID ({mode}) = {result[f'fid_{mode}']}", flush=True)
        torch.cuda.empty_cache()
    if "speed" in metrics:  # memorisation and control work on latents; only the speed test decodes
        rae.to(dev)

    # ---------------------------------------------------------------- memorisation audit (DINOv3 space)
    if "memo" in metrics and pooled:
        gen = torch.nn.functional.normalize(torch.cat(pooled)[:10000].to(dev), dim=1)
        train = torch.nn.functional.normalize(
            torch.from_numpy(np.load(Path(args.cache) / "pooled.npy")[:, 0]).to(dev), dim=1
        )
        nn_sim, nn_idx = [], []
        for i in range(0, len(gen), 1024):
            s = gen[i : i + 1024] @ train.T
            v, ix = s.max(dim=1)
            nn_sim.append(v)
            nn_idx.append(ix)
        nn_sim, nn_idx = torch.cat(nn_sim).cpu(), torch.cat(nn_idx).cpu()
        ref = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:5000].to(dev)]
        s = ref @ train.T
        s[s > 0.9999] = -1  # drop self-matches
        ref_sim = s.max(dim=1).values.cpu()
        thr = float(torch.quantile(ref_sim, 0.999))
        result["memorization"] = {
            "gen_nn_sim_p50": round(float(nn_sim.median()), 4),
            "gen_nn_sim_p99": round(float(torch.quantile(nn_sim, 0.99)), 4),
            "train_nn_sim_p50": round(float(ref_sim.median()), 4),
            "train_nn_sim_p99": round(float(torch.quantile(ref_sim, 0.99)), 4),
            "threshold_train_p99.9": round(thr, 4),
            "frac_gen_above_threshold": round(float((nn_sim > thr).float().mean()), 5),
        }
        print(f"[eval] memorisation: {result['memorization']}", flush=True)
        if need_images:  # closest pairs: generated (top) vs nearest training image (bottom)
            top = nn_sim.topk(8).indices.tolist()
            want = {int(nn_idx[t]): None for t in top}
            idx = 0
            for batch in iter_rows(list_shards(args.data), ["image"], batch_rows=512):
                for cell in batch["image"]:
                    if idx in want:
                        want[idx] = torch.from_numpy(decode_image(cell, 256)).permute(2, 0, 1).float() / 255
                    idx += 1
                if all(v is not None for v in want.values()):
                    break
            gen_imgs = [
                torch.from_numpy(np.asarray(Image.open(img_dir / f"{t:06d}.png"))).permute(2, 0, 1).float()
                / 255
                for t in top
            ]
            save_image(
                torch.stack(gen_imgs + [want[int(nn_idx[t])] for t in top]),
                out / f"{args.tag}_nearest_pairs.png",
                nrow=8,
            )

    # ---------------------------------------------------------------- controllability (no decoding needed)
    if "control" in metrics and Path(args.probe).exists() and attr_names:
        probe = Probe.load(args.probe)
        assert probe.display_names == attr_names, (probe.display_names, attr_names)
        cguid = GuidanceConfig(ig_scale=args.ig_scale, ig_t_min=args.ig_tmin, cfg_scale=args.cfg_scale)
        csample = (
            teacher_sampler(model, cguid, args.steps, args.method)
            if args.kind == "teacher"
            else one_step_sampler(model, args.steps)
        )
        per_attr = {}
        for j, name in enumerate(attr_names):
            accs = {}
            for state, label in ((AttributeTokens.YES, 1), (AttributeTokens.NO, 0)):

                def attrs_fn(i, b, j=j, state=state):
                    a = torch.zeros(b, k, dtype=torch.long)
                    a[:, j] = state
                    return a

                ps = []
                for z, _ in iterate_latents(
                    csample,
                    args.control_n,
                    args.control_batch,
                    k,
                    dev,
                    seed=100 + j,
                    attrs_fn=attrs_fn,
                    autocast=prec.autocast,
                ):
                    ps.append(probe.probs(z.mean(dim=(-2, -1)))[:, j].cpu())
                p = torch.cat(ps)
                accs["yes" if label else "no"] = round(float(((p > 0.5) == bool(label)).float().mean()), 4)
            per_attr[name] = accs
            print(f"[eval] control {name:>14}: yes {accs['yes']:.3f}  no {accs['no']:.3f}", flush=True)
        uncond = torch.cat(pooled) if pooled else None
        result["controllability"] = {
            "cfg_scale": args.cfg_scale,
            "per_attribute": per_attr,
            "mean_accuracy": round(float(np.mean([(v["yes"] + v["no"]) / 2 for v in per_attr.values()])), 4),
            "unconditional_positive_rate": (
                {
                    n: round(float(v), 4)
                    for n, v in zip(
                        attr_names, (probe.probs(uncond.to(dev)) > 0.5).float().mean(0).tolist(), strict=True
                    )
                }
                if uncond is not None
                else None
            ),
        }

    # ---------------------------------------------------------------- speed
    if "speed" in metrics and dev.type == "cuda":
        speed = {}
        for bs in (1, 16):
            noise = torch.randn(bs, 1024, 16, 16, device=dev)
            attrs = torch.zeros(bs, k, dtype=torch.long, device=dev)
            with torch.no_grad(), prec.autocast():
                for _ in range(2):
                    rae.decode(sample_fn(noise, attrs).float())
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                reps = 3
                for _ in range(reps):
                    z = sample_fn(noise, attrs).float()
                torch.cuda.synchronize()
                t_gen = (time.perf_counter() - t0) / reps
                t0 = time.perf_counter()
                for _ in range(reps):
                    rae.decode(z)
                torch.cuda.synchronize()
                t_dec = (time.perf_counter() - t0) / reps
            speed[f"batch{bs}"] = {
                "generator_ms_per_img": round(1000 * t_gen / bs, 2),
                "decoder_ms_per_img": round(1000 * t_dec / bs, 2),
                "images_per_s": round(bs / (t_gen + t_dec), 2),
            }
        result["speed"] = speed
        print(f"[eval] speed: {speed}", flush=True)

    if not args.keep_images and img_dir.exists():
        shutil.rmtree(img_dir)
    (out / f"{args.tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
