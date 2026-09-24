# Transport follows RAEv2 (src/stage2/transport, CC BY-NC 4.0): t=0 is data, t=1 is noise,
# the network predicts x (clean latent), and the loss is measured in velocity space.
"""Flow-matching transport with dimension-dependent timestep shift, plus ODE samplers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

ModelFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (x_t, t) -> x̂ (clean prediction)


def dimension_shift(latent_numel: int, base: int = 4096) -> float:
    """RAEv2: shift = sqrt(dim / 4096); 1024x16x16 latents -> 8."""
    return math.sqrt(latent_numel / base)


def shift_t(t: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * t / (1 + (shift - 1) * t)


@dataclass
class Transport:
    shift: float = 8.0
    t_eps: float = 0.05
    logit_mu: float = 0.0
    logit_sigma: float = 1.0

    def sample_t(self, n: int, device: torch.device) -> torch.Tensor:
        t = torch.sigmoid(torch.randn(n, device=device) * self.logit_sigma + self.logit_mu)
        return shift_t(t, self.shift)

    @staticmethod
    def _bc(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return t.view(-1, *([1] * (x.ndim - 1)))

    def interpolate(self, x1: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        tb = self._bc(t, x1)
        return (1 - tb) * x1 + tb * noise

    def x_to_v(self, x_hat: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (x_t - x_hat) / self._bc(t, x_t).clamp_min(self.t_eps)

    def loss(
        self,
        predict: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        x1: torch.Tensor,
        ig_coeff: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """predict(x_t, t) -> (x_full, x_base). Returns per-sample-mean losses (scalars)."""
        noise = torch.randn_like(x1)
        t = self.sample_t(x1.shape[0], x1.device)
        x_t = self.interpolate(x1, noise, t)
        v_target = self.x_to_v(x1, x_t, t)
        x_full, x_base = predict(x_t, t)
        l_full = (self.x_to_v(x_full.float(), x_t, t) - v_target).pow(2).mean()
        l_base = (self.x_to_v(x_base.float(), x_t, t) - v_target).pow(2).mean()
        return {"loss": l_full + ig_coeff * l_base, "loss_full": l_full, "loss_base": l_base}

    def time_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        return shift_t(torch.linspace(1.0, 0.0, num_steps + 1, device=device), self.shift)

    @torch.no_grad()
    def sample(
        self,
        model_fn: ModelFn,
        noise: torch.Tensor,
        num_steps: int = 50,
        method: str = "euler",
        return_trajectory: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Integrate dx/dt = v from t=1 (noise) to t=0 (data). Heun uses 2 evals/step except the last."""
        ts = self.time_grid(num_steps, noise.device)
        x, traj = noise, []
        for i in range(num_steps):
            t_cur, t_next = ts[i], ts[i + 1]
            tb = t_cur.expand(x.shape[0])
            x_hat = model_fn(x, tb)
            if return_trajectory:
                traj.append(x_hat)
            v = self.x_to_v(x_hat, x, tb)
            x_next = x + (t_next - t_cur) * v
            if method == "heun" and t_next > 0:
                tn = t_next.expand(x.shape[0])
                v2 = self.x_to_v(model_fn(x_next, tn), x_next, tn)
                x_next = x + (t_next - t_cur) * 0.5 * (v + v2)
            x = x_next
        return (x, traj) if return_trajectory else x
