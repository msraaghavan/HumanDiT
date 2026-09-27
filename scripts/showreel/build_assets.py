"""Everything the 15-second reel shows, produced by the models themselves (no stock footage, no hand-typed numbers).

    uv run --extra demo python scripts/showreel/build_assets.py            # -> runs/showreel/assets/
    uv run --extra demo python scripts/showreel/render.py                  # -> docs/assets/showreel.*

Writes, through the demo's own Engine (same samplers, guidance and TrustMark watermark as app/gradio_app.py):
- traj_XX.png      the teacher's clean-latent prediction x̂₀ at each of its 25 ODE steps (hero seed, unconditional)
- edit_X.png       the hero seed with attributes added one at a time (CFG 2.0), plus the probe's verdict on each
- grid_XX.png      the one-step GAN student on seeds 0..71, uncurated
- morph_XX.png     the teacher along a spherical path through noise space between a few seeds
- dcgan.png        one face from the original project's DCGAN sample grid (64 px)
- manifest.json    latent "token views" (PCA of the DINOv3 latent), probe verdicts, live timings on this machine and
                   the measured metrics from runs/eval/*.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
from gradio_app import IG_TUNED, TEACHER_STEPS, Engine  # noqa: E402

from humangen.models.layers import AttributeTokens  # noqa: E402


def load(name: str) -> dict:
    return json.loads((ROOT / "runs" / "eval" / f"{name}.json").read_text())


def token_view(latents: list[torch.Tensor], ref: torch.Tensor) -> list[list[int]]:
    """Project 16x16 grids of 1024-d DINOv3 tokens onto the top 3 principal components of `ref` -> RGB bytes."""
    x = ref[0].flatten(1).T.float().cpu()  # (256, 1024)
    mu = x.mean(0)
    _, _, v = torch.linalg.svd(x - mu, full_matrices=False)
    basis = v[:3].T  # (1024, 3)
    p_ref = (x - mu) @ basis
    lo, hi = p_ref.quantile(0.02, dim=0), p_ref.quantile(0.98, dim=0)
    out = []
    for lat in latents:
        p = (lat[0].flatten(1).T.float().cpu() - mu) @ basis
        rgb = ((p - lo) / (hi - lo)).clamp(0, 1).mul(255).round().to(torch.uint8)
        out.append(rgb.flatten().tolist())  # row-major 16x16, RGB interleaved
    return out


def timed(e: Engine, fn, n: int) -> float:
    ts = []
    for _ in range(n):
        e.sync()
        t0 = time.perf_counter()
        fn()
        e.sync()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def speed(e: Engine) -> dict:
    """Live timings on this machine, measured like the demo's Compare tab: sampling only, batch 1, CFG 2.0."""
    n1, a1 = e.noise(11), e.attrs([])
    for mode, steps in ((e.fast_mode, 1), (e.teacher_mode, TEACHER_STEPS)):  # warm-up both
        e.sample(n1, a1, mode, steps, 2.0, IG_TUNED)
    return {
        "student_ms": round(1000 * timed(e, lambda: e.sample(n1, a1, e.fast_mode, 1, 2.0, IG_TUNED), 20), 1),
        "teacher_s": round(
            timed(e, lambda: e.sample(n1, a1, e.teacher_mode, TEACHER_STEPS, 2.0, IG_TUNED), 5), 2
        ),
        "runs": {"student": 20, "teacher": 5},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--student", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--out", default="runs/showreel/assets")
    ap.add_argument("--hero", type=int, default=4)
    ap.add_argument("--edits", default="Eyeglasses,Smiling,Gray_Hair,Bangs")
    ap.add_argument("--grid", type=int, default=72)
    ap.add_argument("--morph-seeds", default="7,1,9,4")
    ap.add_argument("--morph-frames", type=int, default=24, help="frames per segment")
    ap.add_argument("--speed-only", action="store_true", help="only re-time the samplers into manifest.json")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    e = Engine(str(ROOT / a.teacher), str(ROOT / a.student), str(ROOT / a.probe), [str(ROOT / "runs/eval")])
    if a.speed_only:
        m = json.loads((out / "manifest.json").read_text())
        m["speed"] = speed(e)
        (out / "manifest.json").write_text(json.dumps(m))
        print(m["speed"])
        return
    m: dict = {
        "hero_seed": a.hero,
        "teacher_steps": TEACHER_STEPS,
        "ig": IG_TUNED,
        "gpu": torch.cuda.get_device_name(),
    }

    # 1. the teacher's trajectory for the hero seed, and the latent's token view along it
    noise = e.noise(a.hero)
    z, traj = e.sample(noise, e.attrs([]), e.teacher_mode, TEACHER_STEPS, 1.0, IG_TUNED, trajectory=True)
    for i in range(0, len(traj), 5):
        for j, im in enumerate(e.decode(torch.cat(traj[i : i + 5]).float())):
            im.save(out / f"traj_{i + j:02d}.png")
    m["trajectory_steps"] = len(traj)
    m["tokens"] = {
        "noise": token_view([noise], z)[0],
        "steps": token_view([t.float() for t in traj], z),
    }
    torch.cuda.empty_cache()

    # 2. attributes added one at a time on the same seed; the DINOv3 probe checks each result
    chain = a.edits.split(",")
    m["edits"] = []
    for k in range(len(chain) + 1):
        att = e.attrs([])
        for name in chain[:k]:
            att[:, e.attr_names.index(name)] = AttributeTokens.YES
        ze, _ = e.sample(noise, att, e.teacher_mode, TEACHER_STEPS, 2.0, IG_TUNED)
        e.decode(ze)[0].save(out / f"edit_{k}.png")
        p = e.probs(ze)[0]
        m["edits"].append(
            {"on": chain[:k], "probe": {n: round(float(p[e.attr_names.index(n)]), 3) for n in chain}}
        )
    torch.cuda.empty_cache()

    # 3. the one-step GAN student, seeds 0..N-1 in order (uncurated)
    for s in range(0, a.grid, 8):
        seeds = list(range(s, min(s + 8, a.grid)))
        zg, _ = e.sample(
            torch.cat([e.noise(x) for x in seeds]), e.attrs([], len(seeds)), e.fast_mode, 1, 1.0, IG_TUNED
        )
        for x, im in zip(seeds, e.decode(zg), strict=True):
            im.save(out / f"grid_{x:02d}.png")
    m["grid_seeds"] = [0, a.grid - 1]

    # 4. the teacher along a slerp path through noise space
    ms = [int(s) for s in a.morph_seeds.split(",")]
    frames = []
    for sa, sb in itertools.pairwise(ms):
        za, zb = e.noise(sa), e.noise(sb)
        omega = torch.arccos(((za * zb).sum() / (za.norm() * zb.norm())).clamp(-1 + 1e-6, 1 - 1e-6))
        ts = torch.linspace(0, 1, a.morph_frames + 1, device=e.dev)[:-1]
        frames += [
            (torch.sin((1 - t) * omega) * za + torch.sin(t * omega) * zb) / torch.sin(omega) for t in ts
        ]
    frames.append(e.noise(ms[-1]))
    for i in range(0, len(frames), 4):
        zm, _ = e.sample(
            torch.cat(frames[i : i + 4]),
            e.attrs([], len(frames[i : i + 4])),
            e.teacher_mode,
            TEACHER_STEPS,
            1.0,
            IG_TUNED,
        )
        for j, im in enumerate(e.decode(zm)):
            im.save(out / f"morph_{i + j:02d}.png")
        torch.cuda.empty_cache()
    m["morph"] = {"seeds": ms, "frames": len(frames)}

    m["speed"] = speed(e)

    # 6. the original project's DCGAN (64 px): one tile of its evaluated sample grid (5x5, 2 px padding)
    grid = Image.open(ROOT / "runs/legacy_dcgan/samples.png").convert("RGB")
    r, c = 1, 4
    grid.crop((2 + c * 66, 2 + r * 66, 2 + c * 66 + 64, 2 + r * 66 + 64)).save(out / "dcgan.png")

    # 7. measured metrics, straight from the evaluation JSONs
    legacy, t25, gan = load("legacy_dcgan_64px"), load("teacher_tuned_25step"), load("gan_fixed_1step")
    tuned, ctl = load("teacher_tuned"), load("teacher_tuned_control")
    params = sum(p.numel() for p in e.teacher.parameters())
    m["metrics"] = {
        "fid_before": legacy["fid_clean"],
        "fid_after": t25["fid_clean"],
        "fid_after_n": t25["n"],
        "fid_gan": gan["fid_clean"],
        "precision": t25["precision"],
        "recall": t25["recall"],
        "control_mean": ctl["controllability"]["mean_accuracy"],
        "n_attributes": len(e.attr_names),
        "memorised": tuned["memorization"]["frac_gen_above_threshold"],
        "memo_n": min(tuned["n"], 10000),
        "params_m": round(params / 1e6),
    }
    (out / "manifest.json").write_text(json.dumps(m))
    print(json.dumps({k: v for k, v in m.items() if k != "tokens"}, indent=1))


if __name__ == "__main__":
    main()
