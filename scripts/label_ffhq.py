"""Pseudo-label a (re)built FFHQ latent cache with an already-trained attribute probe, no CelebA needed.

Also a reproducibility check: the label coverage must match the report written when the probe was trained. If the
latents were re-encoded on different hardware (e.g. a TPU), a mismatch means the new latents differ.

    python scripts/label_ffhq.py --probe runs/attributes/attribute_probe.pt --report runs/attributes/attributes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humangen.data.attributes import Probe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument(
        "--report", default="runs/attributes/attributes.json", help="attributes.json from training"
    )
    ap.add_argument("--cache", default="cache/ffhq256")
    ap.add_argument("--tol", type=float, default=0.005, help="max allowed coverage difference per attribute")
    a = ap.parse_args()

    probe = Probe.load(a.probe)
    report = json.loads(Path(a.report).read_text())
    assert report["names"] == probe.display_names, (report["names"], probe.display_names)
    cache = Path(a.cache)
    feats = torch.from_numpy(
        np.load(cache / "pooled.npy")[:, 0]
    )  # un-flipped view, as when the probe was built
    tern = probe.ternary(feats)
    np.save(cache / "attributes.npy", tern)

    worst = 0.0
    coverage = {}
    for j, n in enumerate(probe.display_names):
        c = {s: float((tern[:, j] == v).mean()) for s, v in (("present", 2), ("absent", 1), ("unknown", 0))}
        coverage[n] = c
        ref = report["ffhq_label_coverage"][n]
        diff = max(abs(c[s] - ref[s]) for s in c)
        worst = max(worst, diff)
        print(f"[label] {n:>12}: present {c['present']:.2%} (was {ref['present']:.2%})  max diff {diff:.4f}")
    report["ffhq_label_coverage"] = coverage
    report["relabelled_with_existing_probe"] = {"max_coverage_diff_vs_original": round(worst, 5)}
    (cache / "attributes.json").write_text(json.dumps(report, indent=2))
    print(
        f"[label] wrote {cache / 'attributes.npy'}; max coverage difference vs the original labels: {worst:.4f}"
    )
    if worst > a.tol:
        raise SystemExit(f"coverage differs by {worst:.4f} > {a.tol}: the re-encoded latents do not match")


if __name__ == "__main__":
    main()
