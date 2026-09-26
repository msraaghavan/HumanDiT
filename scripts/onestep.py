"""Train a one-step student from the teacher: W-Flow (2026) or adversarial post-training (GAN).

    python scripts/onestep.py wflow --teacher runs/teacher_b/ckpt/ema.safetensors --minutes 60
    python scripts/onestep.py gan   --teacher runs/teacher_b/ckpt/ema.safetensors --minutes 60
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torchvision.utils import save_image

from humangen.data.latents import LatentDataset, drop_attributes
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.eval.generate import load_generator
from humangen.models.layers import AttributeTokens
from humangen.onestep.adversarial import TrunkDiscriminator, d_loss, g_loss
from humangen.onestep.wflow import grouped_wflow_loss
from humangen.train.trainer import ema_beta, update_ema
from humangen.utils.device import get_precision


class GroupSampler:
    """Condition groups for W-Flow: group 0 is unconditional; others fix one attribute to yes/no."""

    def __init__(self, attrs: np.ndarray, min_count: int = 64, seed: int = 0):
        self.attrs, self.rng = attrs, np.random.default_rng(seed)
        self.n = len(attrs)
        self.options = []
        for j in range(attrs.shape[1]):
            for state in (AttributeTokens.YES, AttributeTokens.NO):
                idx = np.flatnonzero(attrs[:, j] == state)
                if len(idx) >= min_count:
                    self.options.append((j, state, idx))

    def sample(self, groups: int, n_gen: int, n_pos: int, k: int):
        cond, pos_idx = [], []
        for g in range(groups):
            a = np.zeros(k, dtype=np.int64)
            if g == 0 or not self.options:
                pool = None
            else:
                j, state, pool = self.options[self.rng.integers(len(self.options))]
                a[j] = state
            cond.append(np.repeat(a[None], n_gen, 0))
            pos_idx.append(self.rng.integers(self.n, size=n_pos) if pool is None else self.rng.choice(pool, n_pos))
        return torch.from_numpy(np.concatenate(cond)), np.concatenate(pos_idx)


def load_latents(ds: LatentDataset, idx: np.ndarray) -> torch.Tensor:
    return torch.stack([ds[int(i)][0] for i in idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["wflow", "gan"])
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--cache", default="cache/ffhq256")
    ap.add_argument("--out", default=None)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--gen-per-group", type=int, default=16)
    ap.add_argument("--pos-per-group", type=int, default=32)
    ap.add_argument("--gan-batch", type=int, default=64)
    ap.add_argument("--r1-weight", type=float, default=100.0)
    ap.add_argument("--r1-sigma", type=float, default=0.01)
    ap.add_argument("--ema-halflife", type=float, default=20000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    prec = get_precision()
    dev = prec.device
    out = Path(args.out or f"runs/student_{args.method}")
    (out / "samples").mkdir(parents=True, exist_ok=True)
    teacher, attr_names, meta = load_generator(args.teacher, dev)
    k = teacher.cfg.num_attributes
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    # Eager mode (no torch.compile) plus teacher, EMA and AdamW states: batch-128 activations of the 183M model
    # do not fit in 24 GB (OOM on the 4090), so recompute them in the backward pass.
    student.gradient_checkpointing = True
    ema = copy.deepcopy(teacher).eval()
    ds = LatentDataset(args.cache, args.limit)
    attrs_np = ds.attrs if ds.attrs.shape[1] else np.zeros((len(ds), k), dtype=np.int8)
    transport = Transport(shift=dimension_shift(1024 * 16 * 16))
    opt_g = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.0 if args.method == "gan" else 0.9, 0.99),
                              weight_decay=0.0)
    disc = opt_d = None
    if args.method == "gan":
        disc = TrunkDiscriminator(teacher).to(dev)
        opt_d = torch.optim.AdamW(disc.trainable_parameters(), lr=args.d_lr, betas=(0.0, 0.99), weight_decay=0.0)
    groups = GroupSampler(attrs_np, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    bsz = args.groups * args.gen_per_group if args.method == "wflow" else args.gan_batch
    beta = ema_beta(bsz, args.ema_halflife)
    t_one = lambda b: torch.ones(b, device=dev)  # noqa: E731

    def next_batch():
        if args.method == "wflow":
            cond, pidx = groups.sample(args.groups, args.gen_per_group, args.pos_per_group, k)
            return cond, load_latents(ds, pidx), None
        idx = rng.integers(len(ds), size=args.gan_batch)
        real = load_latents(ds, idx)
        real_attrs = torch.from_numpy(attrs_np[idx].astype(np.int64))
        fake_attrs = torch.from_numpy(attrs_np[rng.integers(len(ds), size=args.gan_batch)].astype(np.int64))
        return fake_attrs, real, real_attrs

    pool = ThreadPoolExecutor(1)
    fut = pool.submit(next_batch)
    fixed_noise = torch.randn(16, 1024, 16, 16, generator=torch.Generator().manual_seed(7)).to(dev)
    fixed_attrs = torch.zeros(16, k, dtype=torch.long, device=dev)
    for j in range(8, 16):
        fixed_attrs[j, (j - 8) % k] = AttributeTokens.YES
    log = open(out / "metrics.jsonl", "a")  # noqa: SIM115
    t_start, step = time.perf_counter(), 0
    print(f"[{args.method}] student from {args.teacher} (step {meta.get('step')}), batch {bsz}, "
          f"{len(groups.options)} attribute groups, budget {args.minutes} min", flush=True)

    def save():
        md = {"model_config": meta["model_config"], "attr_names": json.dumps(attr_names), "kind": args.method,
              "step": str(step), "teacher": str(args.teacher)}
        save_file({kk: v.contiguous() for kk, v in ema.state_dict().items()}, str(out / "student.safetensors"), md)

    while step < args.max_steps and time.perf_counter() - t_start < args.minutes * 60:
        cond, real, real_attrs = fut.result()
        fut = pool.submit(next_batch)
        cond, real = cond.to(dev), real.to(dev).float()
        info: dict = {}
        if args.method == "wflow":
            z = torch.randn(len(cond), 1024, 16, 16, device=dev)
            with prec.autocast():
                gen = student(z, t_one(len(z)), cond)[0]
            loss, info = grouped_wflow_loss(gen.float(), real, args.groups)
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            info["loss"] = float(loss)
        else:
            real_attrs = drop_attributes(real_attrs.to(dev), 0.1, 0.3)
            fake_attrs = drop_attributes(cond, 0.1, 0.3)
            z = torch.randn(len(fake_attrs), 1024, 16, 16, device=dev)
            with prec.autocast():  # D step
                with torch.no_grad():
                    fake = student(z, t_one(len(z)), fake_attrs)[0].float()
                ld, info = d_loss(disc, real, real_attrs, fake, fake_attrs, transport, args.r1_weight, args.r1_sigma)
            opt_d.zero_grad(set_to_none=True)
            ld.backward()
            opt_d.step()
            with prec.autocast():  # G step
                fake = student(z, t_one(len(z)), fake_attrs)[0].float()
                lg = g_loss(disc, fake, fake_attrs, transport)
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            info.update(loss_d=float(ld), loss_g=float(lg))
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_g.step()
        update_ema(ema, student, beta)
        step += 1
        if step % 25 == 0 or step == 1:
            rec = {"step": step, "minutes": round((time.perf_counter() - t_start) / 60, 2), "grad_norm": float(gn), **info}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            main_loss = info.get("loss", info.get("loss_g"))
            mem = f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB" if dev.type == "cuda" else ""
            print(f"[{args.method}] step {step} {rec['minutes']} min loss {main_loss:.4f} gnorm {float(gn):.2f}{mem}",
                  flush=True)
            if not math.isfinite(main_loss):
                raise FloatingPointError("non-finite loss")
        if step % args.sample_every == 0:
            with torch.no_grad(), prec.autocast():
                zs = ema(fixed_noise, t_one(16), fixed_attrs)[0].float()
            torch.save(zs.cpu(), out / "samples" / f"step{step:06d}.pt")
            save()
    save()
    with torch.no_grad(), prec.autocast():
        zs = ema(fixed_noise, t_one(16), fixed_attrs)[0].float()
    torch.save(zs.cpu(), out / "samples" / "final.pt")
    if args.decode:
        from humangen.rae import RAE

        rae = RAE(load_encoder=False).to(dev)
        save_image(rae.decode(zs), out / "samples" / "final.png", nrow=8, padding=2)
    (out / "summary.json").write_text(json.dumps({"method": args.method, "steps": step,
                                                  "minutes": round((time.perf_counter() - t_start) / 60, 2),
                                                  "batch": bsz, "teacher": args.teacher}, indent=2))
    print(f"[{args.method}] done: {step} steps -> {out / 'student.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
