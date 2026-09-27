"""Why did the one-step GAN lose facial hair? Probe-scored attribute control under different request patterns.

    uv run python scripts/control_diagnosis.py --out runs/eval_local/control_diagnosis.json

For each model, generates n latents per request and reports how often the DINOv3 probe finds each requested
attribute (and Male). A request is a set of attributes asked *present*; every other attribute is left unspecified,
as in scripts/evaluate.py --metrics control. No decoding: the probe reads latents.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from humangen.data.attributes import Probe
from humangen.diffusion.guidance import GuidanceConfig
from humangen.eval.generate import load_generator, one_step_sampler, teacher_sampler
from humangen.models.layers import AttributeTokens
from humangen.utils.device import get_precision

REQUESTS = {
    "unconditional": [],
    "Goatee": ["Goatee"],
    "Sideburns": ["Sideburns"],
    "Beard": ["Beard"],
    "Eyeglasses": ["Eyeglasses"],
    "Male": ["Male"],
    "Goatee+Male": ["Goatee", "Male"],
    "Sideburns+Male": ["Sideburns", "Male"],
    "Beard+Male": ["Beard", "Male"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gan", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--wflow", default="runs/student_wflow/student.safetensors")
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--teacher-steps", type=int, default=25)
    ap.add_argument(
        "--teacher-requests",
        default="unconditional,Goatee,Beard,Male,Goatee+Male",
        help="the teacher is ~25x slower per sample, so it runs on a subset",
    )
    ap.add_argument("--out", default="runs/eval_local/control_diagnosis.json")
    a = ap.parse_args()
    prec = get_precision()
    dev = prec.device
    probe = Probe.load(a.probe)
    names = probe.display_names
    t_req = a.teacher_requests.split(",")
    # (label, checkpoint, sampler factory, requests)
    setups = [
        ("GAN (fixed), 1 step", a.gan, lambda m: one_step_sampler(m, 1), list(REQUESTS)),
        ("W-Flow, 1 step", a.wflow, lambda m: one_step_sampler(m, 1), list(REQUESTS)),
        (
            "teacher, IG 1.78, no CFG (the GAN's anchor setting)",
            a.teacher,
            lambda m: teacher_sampler(m, GuidanceConfig(), a.teacher_steps),
            t_req,
        ),
        (
            "teacher, IG 1.55, CFG 2.0 (the control table's setting)",
            a.teacher,
            lambda m: teacher_sampler(m, GuidanceConfig(ig_scale=1.55, cfg_scale=2.0), a.teacher_steps),
            t_req,
        ),
    ]
    result: dict = {"n": a.n, "teacher_steps": a.teacher_steps, "models": {}}
    for label, ckpt, make, reqs in setups:
        model, attr_names, _ = load_generator(ckpt, dev)
        assert attr_names == names, (attr_names, names)
        sample = make(model)
        rows = {}
        for req in reqs:
            t0 = time.perf_counter()
            ps = []
            g = torch.Generator().manual_seed(1234)  # same noise for every model and request
            for i in range(0, a.n, a.batch):
                b = min(a.batch, a.n - i)
                attrs = torch.zeros(b, len(names), dtype=torch.long)
                for n_ in REQUESTS[req]:
                    attrs[:, names.index(n_)] = AttributeTokens.YES
                noise = torch.randn(b, 1024, 16, 16, generator=g)
                with torch.no_grad(), prec.autocast():
                    z = sample(noise.to(dev), attrs.to(dev)).float()
                ps.append(probe.probs(z.mean(dim=(-2, -1))).cpu())
            p = torch.cat(ps) > 0.5
            shown = sorted({*REQUESTS[req], "Male", "Goatee", "Sideburns", "Beard"}, key=names.index)
            rows[req] = {n_: round(float(p[:, names.index(n_)].float().mean()), 3) for n_ in shown}
            print(f"[diag] {label:<55} {req:<16} {rows[req]}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        result["models"][label] = rows
        del model, sample
        gc.collect()
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"[diag] wrote {a.out}")


if __name__ == "__main__":
    main()
