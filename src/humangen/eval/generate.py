"""Load trained generators and produce latents / images in batches."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from humangen.diffusion.guidance import GuidanceConfig, guided_model_fn
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.models.dit_dh import DiTDHConfig, HumanDiTDH


def load_generator(path: str | Path, device: torch.device) -> tuple[HumanDiTDH, list[str], dict]:
    """Load an EMA/student checkpoint written as safetensors with model_config metadata."""
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
    cfg = DiTDHConfig(**json.loads(meta["model_config"]))
    model = HumanDiTDH(cfg)
    model.load_state_dict(load_file(str(path)))
    return model.to(device).eval().requires_grad_(False), json.loads(meta.get("attr_names", "[]")), meta


def teacher_sampler(
    model: HumanDiTDH, guidance: GuidanceConfig, num_steps: int = 50, method: str = "euler"
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    transport = Transport(shift=dimension_shift(1024 * 16 * 16))

    def sample(noise: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        return transport.sample(guided_model_fn(model, attrs, guidance), noise, num_steps, method)

    return sample


def one_step_sampler(
    model: HumanDiTDH, steps: int = 1
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Students predict x̂ from noise at t=1; for steps > 1, re-noise to the next t and predict again."""
    transport = Transport(shift=dimension_shift(1024 * 16 * 16))

    @torch.no_grad()
    def sample(noise: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
        ts = transport.time_grid(steps, noise.device)
        x = noise
        for i in range(steps):
            t = ts[i].expand(noise.shape[0])
            x_hat = model(x, t, attrs)[0].float()
            if i + 1 < steps:
                x = transport.interpolate(x_hat, torch.randn_like(x_hat), ts[i + 1].expand(noise.shape[0]))
            else:
                x = x_hat
        return x

    return sample


def iterate_latents(
    sample_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n: int,
    batch: int,
    num_attributes: int,
    device: torch.device,
    seed: int = 0,
    attrs_fn: Callable[[int, int], torch.Tensor] | None = None,
    autocast: Callable | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yields (latents, attrs) batches with deterministic per-batch noise."""
    for i in range(0, n, batch):
        b = min(batch, n - i)
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + i)
        noise = torch.randn(b, 1024, 16, 16, generator=g).to(device)
        attrs = (attrs_fn(i, b) if attrs_fn else torch.zeros(b, num_attributes, dtype=torch.long)).to(device)
        ctx = autocast() if autocast else torch.no_grad()
        with torch.no_grad(), ctx:
            yield sample_fn(noise, attrs).float(), attrs
