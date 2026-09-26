"""Teacher training loop: flow matching + Internal Guidance on RAE latents, CMuon, EMA, resume, budget fit."""

from __future__ import annotations

import copy
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from humangen.data.latents import LatentDataset, drop_attributes
from humangen.diffusion.guidance import GuidanceConfig, guided_model_fn
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.models.dit_dh import HumanDiTDH, count_params, make_config
from humangen.models.layers import AttributeTokens
from humangen.optim.param_groups import build_optimizer
from humangen.utils.device import get_precision


def lr_factor(step: int, total: int, const_frac: float, decay_end_frac: float, final_ratio: float) -> float:
    """RAEv2 schedule shape: constant, then linear decay to final_ratio, then constant."""
    a, b = const_frac * total, decay_end_frac * total
    if step < a:
        return 1.0
    if step >= b:
        return final_ratio
    return 1.0 - (1.0 - final_ratio) * (step - a) / max(b - a, 1)


def ema_beta(batch_size: int, halflife_samples: float) -> float:
    return 0.5 ** (batch_size / halflife_samples)


@torch.no_grad()
def update_ema(ema: torch.nn.Module, model: torch.nn.Module, beta: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
        pe.lerp_(pm.detach().float(), 1.0 - beta)


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.out = Path(cfg.out_dir) / cfg.run_name
        (self.out / "ckpt").mkdir(parents=True, exist_ok=True)
        (self.out / "samples").mkdir(exist_ok=True)
        torch.manual_seed(cfg.seed)
        self.prec = get_precision(cfg.train.precision if cfg.train.precision != "auto" else None)
        self.dev = self.prec.device

        self.data = LatentDataset(cfg.data.cache, cfg.data.limit, in_memory=cfg.data.in_memory)
        k = self.data.attrs.shape[1]
        self.attr_names = self.data.attr_names
        mcfg = make_config(
            cfg.model.preset, num_attributes=max(k, 1), **dict(cfg.model.get("overrides") or {})
        )
        self.has_attrs = k > 0
        self.model = HumanDiTDH(mcfg).to(self.dev)
        self.model.gradient_checkpointing = bool(cfg.model.grad_checkpointing)
        self.ema = copy.deepcopy(self.model).requires_grad_(False).eval()
        self.opt = build_optimizer(self.model, cfg.train.optimizer, cfg.train.lr)
        self.transport = Transport(shift=dimension_shift(1024 * 16 * 16))
        self.fwd = torch.compile(self.model) if cfg.train.compile else self.model
        self.step, self.samples_seen, self.total_steps = 0, 0, int(cfg.train.max_steps)
        self.writer = SummaryWriter(str(self.out / "tb"))
        self.metrics_f = open(self.out / "metrics.jsonl", "a")  # noqa: SIM115
        self._resume()
        print(
            f"[train] {cfg.run_name}: {count_params(self.model) / 1e6:.1f}M params, precision={self.prec.name}, "
            f"attrs={self.attr_names or 'none'}, data={len(self.data)} images x{self.data.flips} views",
            flush=True,
        )

    # ------------------------------------------------------------------ state
    def _state(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "step": self.step,
            "samples_seen": self.samples_seen,
            "total_steps": self.total_steps,
            "model_config": self.model.cfg.to_dict(),
            "attr_names": self.attr_names,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save(self) -> None:
        _atomic_save(self._state(), self.out / "ckpt" / "latest.pt")
        meta = {
            "model_config": json.dumps(self.model.cfg.to_dict()),
            "attr_names": json.dumps(self.attr_names),
            "step": str(self.step),
            "samples_seen": str(self.samples_seen),
        }
        save_file(
            {k: v.contiguous() for k, v in self.ema.state_dict().items()},
            str(self.out / "ckpt" / "ema.safetensors"),
            meta,
        )

    def _resume(self) -> None:
        path = self.out / "ckpt" / "latest.pt"
        if not path.exists():
            return
        st = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(st["model"])
        self.ema.load_state_dict(st["ema"])
        self.opt.load_state_dict(st["opt"])
        self.step, self.samples_seen, self.total_steps = st["step"], st["samples_seen"], st["total_steps"]
        torch.set_rng_state(st["rng"]["torch"])
        if st["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])
        print(f"[train] resumed from step {self.step} ({self.samples_seen} samples)", flush=True)

    # ------------------------------------------------------------------ sampling
    def sample_grid(self, tag: str) -> Path | None:
        """Fixed-noise samples from the EMA model; decoded on cfg.log.decoder_device."""
        n = 16
        g = torch.Generator(device="cpu").manual_seed(1234)
        noise = torch.randn(n, 1024, 16, 16, generator=g).to(self.dev)
        k = self.model.cfg.num_attributes
        attrs = torch.zeros(n, k, dtype=torch.long, device=self.dev)
        if self.has_attrs:  # rows 2-4: one attribute switched on per column group
            for j in range(8, n):
                attrs[j, (j - 8) % k] = AttributeTokens.YES
        fn = guided_model_fn(self.ema, attrs, GuidanceConfig(cfg_scale=2.0 if self.has_attrs else 1.0))
        with torch.no_grad(), self.prec.autocast():
            z = self.transport.sample(fn, noise, num_steps=self.cfg.log.sample_steps)
        torch.save(z.cpu(), self.out / "samples" / f"{tag}.pt")
        if not self.cfg.log.decode_samples:
            return None
        from torchvision.utils import save_image

        from humangen.rae import RAE

        rae = RAE(load_encoder=False).to(self.cfg.log.decoder_device)
        imgs = []
        with torch.no_grad():
            for i in range(0, n, 4):
                imgs.append(rae.decode(z[i : i + 4].to(self.cfg.log.decoder_device)).cpu())
        del rae
        path = self.out / "samples" / f"{tag}.png"
        save_image(torch.cat(imgs), path, nrow=8, padding=2)
        return path

    # ------------------------------------------------------------------ budget
    def fit_budget(self, step_seconds: float) -> None:
        b = self.cfg.budget
        hours = []
        if b.get("budget_usd") and b.get("usd_per_hour"):
            hours.append(b.budget_usd / b.usd_per_hour - b.get("reserve_hours", 0.0))
        if b.get("max_train_hours"):
            hours.append(b.max_train_hours)
        if not hours:
            return
        affordable = int(max(min(hours), 0) * 3600 / step_seconds)
        if affordable < self.total_steps:
            print(
                f"[budget] {self.total_steps} steps would take {self.total_steps * step_seconds / 3600:.2f} h; "
                f"fitting to {affordable} steps ({min(hours):.2f} h)",
                flush=True,
            )
            self.total_steps = affordable
        cost = self.total_steps * step_seconds / 3600 * (b.get("usd_per_hour") or 0)
        print(
            f"[budget] plan: {self.total_steps} steps x {self.cfg.train.batch_size} = "
            f"{self.total_steps * self.cfg.train.batch_size / 1e6:.2f}M samples, "
            f"~{self.total_steps * step_seconds / 3600:.2f} h, ~${cost:.2f}",
            flush=True,
        )

    # ------------------------------------------------------------------ loop
    def _loader(self, micro: int):
        cfg = self.cfg
        return iter(
            DataLoader(
                self.data,
                batch_size=micro,
                sampler=RandomSampler(self.data, replacement=True, num_samples=10**9),
                num_workers=cfg.data.num_workers,
                pin_memory=self.dev.type == "cuda",
                drop_last=True,
                persistent_workers=cfg.data.num_workers > 0,
                prefetch_factor=4 if cfg.data.num_workers > 0 else None,
            )
        )

    def _micro_step(self, it, micro: int, n_micro: int, tc) -> dict[str, torch.Tensor]:
        """Accumulate gradients over n_micro micro-batches; returns averaged (detached) losses."""
        acc: dict[str, torch.Tensor] = {}
        for _ in range(n_micro):
            x1, attrs = next(it)
            x1 = x1.to(self.dev, non_blocking=True).float()
            attrs = attrs.to(self.dev, non_blocking=True)
            if attrs.shape[1] == 0:
                attrs = torch.zeros(x1.shape[0], 1, dtype=torch.long, device=self.dev)
            attrs = drop_attributes(attrs, tc.cfg_drop, tc.attr_drop)
            with self.prec.autocast():
                losses = self.transport.loss(lambda xt, t, a=attrs: self.fwd(xt, t, a), x1, tc.ig_coeff)
            (losses["loss"] / n_micro).backward()
            for k, v in losses.items():
                acc[k] = acc.get(k, 0) + v.detach() / n_micro
        return acc

    def train(self) -> None:
        cfg, tc = self.cfg, self.cfg.train
        micro = int(tc.micro_batch) if str(tc.get("micro_batch", "auto")) != "auto" else tc.batch_size
        beta = ema_beta(tc.batch_size, tc.ema_halflife_samples)
        final_ratio = tc.final_lr / tc.lr
        it = self._loader(micro)
        t_start, t_log, last_ckpt = time.perf_counter(), time.perf_counter(), time.perf_counter()
        timing, budget_fitted = [], self.step > 0
        self.model.train()
        while self.step < self.total_steps:
            t0 = time.perf_counter()
            f = lr_factor(self.step, self.total_steps, tc.const_frac, tc.decay_end_frac, final_ratio)
            for group in self.opt.param_groups:
                group["lr"] = tc.lr * f
            self.opt.zero_grad(set_to_none=True)
            try:
                losses = self._micro_step(it, micro, tc.batch_size // micro, tc)
            except torch.OutOfMemoryError:
                if micro <= 1 or self.step > 0:
                    raise
                micro //= 2
                self.opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(
                    f"[train] OOM at micro-batch {micro * 2}; retrying with {micro} x {tc.batch_size // micro}",
                    flush=True,
                )
                it = self._loader(micro)
                continue
            except Exception as e:  # torch.compile failures surface on the first call
                if self.step > 0 or self.fwd is self.model:
                    raise
                print(
                    f"[train] torch.compile failed ({type(e).__name__}: {str(e)[:200]}); using eager",
                    flush=True,
                )
                self.fwd = self.model
                self.opt.zero_grad(set_to_none=True)
                continue
            gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.clip_grad)
            self.opt.step()
            update_ema(self.ema, self.model, beta)
            self.step += 1
            self.samples_seen += tc.batch_size

            if self.dev.type == "cuda" and (not budget_fitted or self.step % cfg.log.every == 0):
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if not budget_fitted:
                timing.append(dt)
                if len(timing) == 30:  # skip compile/warm-up steps, then fit the run to the budget
                    self.fit_budget(float(np.median(timing[10:])))
                    budget_fitted = True

            if self.step % cfg.log.every == 0 or self.step == 1:
                elapsed = time.perf_counter() - t_log
                t_log = time.perf_counter()
                ips = cfg.log.every * tc.batch_size / elapsed if self.step > 1 else 0.0
                rec = {
                    "step": self.step,
                    "samples": self.samples_seen,
                    "lr": tc.lr * f,
                    "grad_norm": float(gnorm),
                    "img_per_s": round(ips, 1),
                    "hours": round((time.perf_counter() - t_start) / 3600, 3),
                    **{k: v.item() for k, v in losses.items()},
                }
                if self.dev.type == "cuda":
                    rec["mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                for k2, v in rec.items():
                    if k2 != "step":
                        self.writer.add_scalar(f"train/{k2}", v, self.step)
                self.metrics_f.write(json.dumps(rec) + "\n")
                self.metrics_f.flush()
                eta = (
                    (self.total_steps - self.step) * tc.batch_size / max(ips, 1e-6) / 3600
                    if ips
                    else float("nan")
                )
                print(
                    f"[train] step {self.step}/{self.total_steps} loss {rec['loss']:.4f} "
                    f"(full {rec['loss_full']:.4f} base {rec['loss_base']:.4f}) gnorm {rec['grad_norm']:.2f} "
                    f"{ips:.1f} img/s eta {eta:.2f} h",
                    flush=True,
                )
                if not math.isfinite(rec["loss"]):
                    raise FloatingPointError("non-finite loss")

            if self.step % cfg.log.sample_every == 0:
                self.model.eval()
                p = self.sample_grid(f"step{self.step:07d}")
                self.model.train()
                print(f"[train] samples -> {p or 'latents only'}", flush=True)
            if time.perf_counter() - last_ckpt > cfg.log.ckpt_every_min * 60:
                self.save()
                last_ckpt = time.perf_counter()

        self.save()
        self.model.eval()
        p = self.sample_grid(f"final_step{self.step:07d}")
        (self.out / "train_summary.json").write_text(
            json.dumps(
                {
                    "steps": self.step,
                    "samples_seen": self.samples_seen,
                    "hours": round((time.perf_counter() - t_start) / 3600, 3),
                    "params_m": round(count_params(self.model) / 1e6, 1),
                    "precision": self.prec.name,
                    "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "platform": platform.platform(),
                    "final_samples": str(p),
                },
                indent=2,
            )
        )
        print(f"[train] done: {self.step} steps; final samples -> {p}", flush=True)
