"""Split HumanDiT^DH parameters between CMuon and AdamW, following the CMuon paper.

CMuon: 2D hidden weights of attention / MLP / adaLN / projector layers.
  - adaLN modulations are fused projections: 6 chunks in head blocks, 2 in final layers' adaLN.
  - q/k/v and SwiGLU gate/up are already separate Linear layers here (as in RAEv2), so 1 chunk.
AdamW: 1D params (norms, biases), embeddings (latent/time/attribute embedders, learnable tokens),
and the output projections of the final layers.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer

from humangen.optim.cmuon import CMuon

ADAMW_PREFIXES = ("s_embedder.", "x_embedder.", "t_embedder.", "attr_embedder.")
ADAMW_EXACT = ("final_layer.linear.weight", "base_final_layer.linear.weight")


def split_params(model: nn.Module) -> tuple[dict[int, list[nn.Parameter]], list[nn.Parameter], list[str]]:
    """Returns ({chunks: [params]} for CMuon, [params] for AdamW, human-readable report)."""
    muon: dict[int, list[nn.Parameter]] = {}
    adamw: list[nn.Parameter] = []
    report: list[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim != 2 or name.startswith(ADAMW_PREFIXES) or name in ADAMW_EXACT:
            adamw.append(p)
            continue
        chunks = 1
        if "adaln_modulation" in name:
            chunks = p.shape[0] // p.shape[1]  # 6 for head blocks, 2 for final layers
        muon.setdefault(chunks, []).append(p)
        report.append(f"cmuon[{chunks}] {name} {tuple(p.shape)}")
    return muon, adamw, report


class CMuonAdamW(Optimizer):
    """Composite optimizer stepping CMuon (2D hidden weights) and AdamW (everything else) together."""

    def __init__(self, muon: CMuon, adamw: torch.optim.AdamW):
        self.muon, self.adamw = muon, adamw
        self.param_groups = muon.param_groups + adamw.param_groups
        self.defaults = {}
        self.state = {}  # per-optimizer state lives in the children

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none)
        self.adamw.zero_grad(set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def build_optimizer(
    model: nn.Module, kind: str = "cmuon", lr: float = 2e-4, betas: Iterable[float] = (0.9, 0.95)
) -> Optimizer:
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(betas), weight_decay=0.0)
    muon_groups, adamw_params, _ = split_params(model)
    if kind == "muon":  # plain Muon ablation: no chunking
        muon_groups = {1: [p for ps in muon_groups.values() for p in ps]}
    muon = CMuon([{"params": ps, "chunks": c} for c, ps in muon_groups.items()], lr=lr)
    fused = torch.cuda.is_available() and all(p.is_cuda for p in adamw_params)
    adamw = torch.optim.AdamW(adamw_params, lr=lr, betas=tuple(betas), weight_decay=0.0, fused=fused)
    return CMuonAdamW(muon, adamw)
