"""Gate before GAN training: can the discriminator see a requested attribute that is missing?

    python scripts/check_discriminator.py --teacher runs/teacher_b/ckpt/ema.safetensors --cache cache/ffhq256

The anchored GAN student lost facial hair while its discriminator sat at chance (scripts/control_diagnosis.py). If D
cannot tell a face under its own attributes from the same face with one attribute flipped, it gives the student no
reason to draw what was requested. This trains only D's heads, exactly as in scripts/onestep.py but with the student
frozen at its initialisation, for the original D and for the attribute-aware one. It then scores held-out real
latents: for each attribute, how often D prefers the true condition over the same condition with that attribute
flipped. Exits 1 (so the Colab driver skips the student run) unless the attribute-aware D reaches --min-acc on the
facial-hair attributes. No generator is trained.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from humangen.data.latents import LatentDataset, drop_attributes
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.eval.generate import load_generator
from humangen.models.layers import AttributeTokens
from humangen.onestep.adversarial import TrunkDiscriminator, d_loss
from humangen.utils.device import get_precision

WATCH = ("Goatee", "Sideburns", "Beard")
CHUNK = 128


def load(ds: LatentDataset, idx: np.ndarray) -> torch.Tensor:
    return torch.stack([ds[int(i)][0] for i in idx])


def score(disc: TrunkDiscriminator, x: torch.Tensor, t: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [disc(x[i : i + CHUNK], t[i : i + CHUNK], attrs[i : i + CHUNK]) for i in range(0, len(x), CHUNK)]
    ).float()


def flip(attrs: torch.Tensor, j: int) -> torch.Tensor:
    out = attrs.clone()
    yes = attrs[:, j] == AttributeTokens.YES
    out[:, j] = torch.where(
        yes,
        torch.full_like(attrs[:, j], AttributeTokens.NO),
        torch.full_like(attrs[:, j], AttributeTokens.YES),
    )
    return out


@torch.no_grad()
def evaluate(disc, teacher, ev_x, ev_a, names, transport, t_val: float) -> dict:
    dev = ev_x.device
    g = torch.Generator(device=dev).manual_seed(1)
    noise = torch.randn(ev_x.shape, device=dev, generator=g)
    t = torch.full((len(ev_x),), t_val, device=dev)
    xt = transport.interpolate(ev_x, noise, t)
    s_true = score(disc, xt, t, ev_a)
    per = {}
    for j, name in enumerate(names):
        rows = ev_a[:, j] != AttributeTokens.NULL
        if rows.sum() >= 16:
            s_flip = score(disc, xt, t, flip(ev_a, j))
            per[name] = round(prefer(s_true[rows], s_flip[rows]), 3)
    # the student at initialisation (the teacher, one step from t = 1) under the same conditions and noise level
    zf = torch.randn(ev_x.shape, device=dev, generator=g)
    ones = torch.ones(len(zf), device=dev)
    fake = torch.cat(
        [
            teacher(zf[i : i + CHUNK], ones[i : i + CHUNK], ev_a[i : i + CHUNK])[0]
            for i in range(0, len(zf), CHUNK)
        ]
    ).float()
    s_fake = score(disc, transport.interpolate(fake, noise, t), t, ev_a)
    return {
        "flip_accuracy": per,
        "facial_hair_flip_accuracy": round(float(np.mean([per[n] for n in WATCH if n in per])), 3),
        "all_attributes_flip_accuracy": round(float(np.mean(list(per.values()))), 3),
        "real_vs_fake_accuracy": round(prefer(s_true, s_fake), 3),
    }


def prefer(a: torch.Tensor, b: torch.Tensor) -> float:
    """How often a scores above b; ties count half (a D that cannot see an attribute gives ties, not wins)."""
    return float(((a > b).float() + 0.5 * (a == b).float()).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--cache", default="cache/ffhq256")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--heldout", type=int, default=2048)
    ap.add_argument(
        "--min-acc", type=float, default=0.75, help="facial-hair flip accuracy the new D must reach"
    )
    ap.add_argument("--out", default="runs/eval/discriminator_check.json")
    a = ap.parse_args()
    prec = get_precision()
    dev = prec.device
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    teacher, names, _ = load_generator(a.teacher, dev)
    ds = LatentDataset(a.cache)
    attrs_np = ds.attrs.astype(np.int64)
    n_train = len(ds) - a.heldout
    ev_idx = np.arange(n_train, len(ds))
    ev_x, ev_a = load(ds, ev_idx).to(dev).float(), torch.from_numpy(attrs_np[ev_idx]).to(dev)
    transport = Transport(shift=dimension_shift(1024 * 16 * 16))
    result: dict = {"steps": a.steps, "batch": a.batch, "heldout": len(ev_idx), "variants": {}}

    for label, attr_heads, mismatch in (("original D", False, 0.0), ("attribute-aware D", True, 1.0)):
        t0 = time.perf_counter()
        disc = TrunkDiscriminator(teacher, attr_heads=attr_heads).to(dev)
        opt = torch.optim.AdamW(disc.trainable_parameters(), lr=1e-4, betas=(0.0, 0.99), weight_decay=0.0)
        for _ in range(
            a.steps
        ):  # D steps as in onestep.py; the student stays at its initialisation (the teacher)
            idx = rng.integers(n_train, size=a.batch)
            real = load(ds, idx).to(dev).float()
            real_attrs = drop_attributes(torch.from_numpy(attrs_np[idx]).to(dev), 0.1, 0.3)
            fake_attrs = drop_attributes(
                torch.from_numpy(attrs_np[rng.integers(n_train, size=a.batch)]).to(dev), 0.1, 0.3
            )
            with prec.autocast():
                with torch.no_grad():
                    z = torch.randn(a.batch, 1024, 16, 16, device=dev)
                    fake = teacher(z, torch.ones(a.batch, device=dev), fake_attrs)[0].float()
                ld, _ = d_loss(disc, real, real_attrs, fake, fake_attrs, transport, 100.0, 0.01, mismatch)
            opt.zero_grad(set_to_none=True)
            ld.backward()
            opt.step()
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # score in full fp32: bf16/TF32 rounding would create ties
        )
        scores = {f"t={t}": evaluate(disc, teacher, ev_x, ev_a, names, transport, t) for t in (0.1, 0.4)}
        torch.backends.cuda.matmul.allow_tf32 = tf32
        result["variants"][label] = {**scores, "seconds": round(time.perf_counter() - t0, 1)}
        print(f"[dcheck] {label}: {json.dumps(scores)}", flush=True)
        del disc, opt
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    new = result["variants"]["attribute-aware D"]
    acc = min(new[t]["facial_hair_flip_accuracy"] for t in ("t=0.1", "t=0.4"))
    result["min_acc"] = a.min_acc
    result["passed"] = a.min_acc <= 0 or bool(acc >= a.min_acc)  # --min-acc 0: rehearsal, no gate
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    verdict = "PASS" if result["passed"] else "FAIL -- not training the student"
    print(
        f"[dcheck] attribute-aware D, facial-hair flip accuracy (worst t) {acc:.3f}, need {a.min_acc}: {verdict}"
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
