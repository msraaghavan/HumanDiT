"""Train the DINOv3 attribute probe on CelebA latents and pseudo-label FFHQ.

Requires `prepare_data.py celeba` and `prepare_data.py ffhq` caches. Writes:
    cache/attribute_probe.pt                 probe restricted to the selected attributes
    cache/ffhq256/attributes.npy / .json     (N, K) int8 {0 unknown, 1 absent, 2 present} + report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from humangen.data.attributes import (
    Probe,
    calibrate_abstention,
    evaluate,
    fit_linear_probe,
    select_attributes,
    write_report,
)


def load_split(cache: Path, name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = cache / name
    man = json.loads((d / "manifest.json").read_text())
    return np.load(d / "pooled.npy")[:, 0], np.load(d / "labels.npy"), man["label_names"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--min-bacc", type=float, default=0.88)
    ap.add_argument("--max-k", type=int, default=12)
    ap.add_argument("--min-count", type=int, default=20, help="min support for abstention thresholds")
    ap.add_argument("--min-precision", type=float, default=0.90)
    ap.add_argument("--min-npv", type=float, default=0.97)
    ap.add_argument("--min-ffhq-frac", type=float, default=0.005,
                    help="drop attributes with fewer FFHQ faces than this labelled present (or absent)")
    args = ap.parse_args()
    cache = Path(args.cache)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    xtr, ytr, names = load_split(cache, "celeba_train")
    xva, yva, _ = load_split(cache, "celeba_valid")
    xte, yte, _ = load_split(cache, "celeba_test")
    print(f"[attrs] CelebA train {len(xtr)} / valid {len(xva)} / test {len(xte)}; {len(names)} attributes", flush=True)

    w, b, mu, sigma = fit_linear_probe(xtr, ytr, xva, yva, device=dev)
    k_all = len(names)
    full = Probe(names, w, b, mu, sigma, np.zeros(k_all), np.ones(k_all))
    test_metrics = evaluate(full, xte, yte)
    candidates = select_attributes(test_metrics, args.min_bacc, len(names))  # best first
    cidx = [names.index(n) for n in candidates]
    p_val = torch.sigmoid(full.logits(torch.from_numpy(xva))).numpy()[:, cidx]
    lo, hi = calibrate_abstention(p_val, yva[:, cidx], args.min_precision, args.min_npv, args.min_count)
    cand = Probe(candidates, w[cidx], b[cidx], mu, sigma, lo, hi)

    # The teacher can only learn an attribute it sees in both states, so drop attributes that are (almost)
    # never labelled present or absent on FFHQ, and back-fill from the next-best candidates.
    ffhq = cache / "ffhq256"
    feats = torch.from_numpy(np.load(ffhq / "pooled.npy")[:, 0])  # un-flipped view; attributes are flip-invariant
    tern_all = cand.ternary(feats)
    keep, dropped = [], {}
    for j, n in enumerate(cand.display_names):
        pres, absn = float((tern_all[:, j] == 2).mean()), float((tern_all[:, j] == 1).mean())
        if min(pres, absn) >= args.min_ffhq_frac and len(keep) < args.max_k:
            keep.append(j)
        elif min(pres, absn) < args.min_ffhq_frac:
            dropped[n] = {"present": pres, "absent": absn}
    chosen = [candidates[j] for j in keep]
    idx = [cidx[j] for j in keep]
    probe = Probe(chosen, w[idx], b[idx], mu, sigma, lo[keep], hi[keep])
    probe.save(cache / "attribute_probe.pt")
    tern = tern_all[:, keep]
    np.save(ffhq / "attributes.npy", tern)
    coverage = {n: {"present": float((tern[:, j] == 2).mean()), "absent": float((tern[:, j] == 1).mean()),
                    "unknown": float((tern[:, j] == 0).mean())} for j, n in enumerate(probe.display_names)}
    report = {
        "names": probe.display_names,
        "celeba_columns": chosen,
        "selection": {"min_balanced_acc": args.min_bacc, "max_k": args.max_k, "min_ffhq_frac": args.min_ffhq_frac,
                      "dropped_for_ffhq_coverage": dropped},
        "abstention": {"min_precision": args.min_precision, "min_npv": args.min_npv,
                       "lo": lo[keep].round(4).tolist(), "hi": hi[keep].round(4).tolist()},
        "celeba_test_metrics_all": test_metrics,
        "ffhq_label_coverage": coverage,
    }
    write_report(ffhq / "attributes.json", report)
    print(f"[attrs] selected {len(chosen)}: " + ", ".join(
        f"{probe.display_names[j]} (bacc {test_metrics[c]['balanced_acc']:.3f})" for j, c in enumerate(chosen)))
    for n, c in dropped.items():
        print(f"[attrs] dropped {n}: FFHQ present {c['present']:.2%} / absent {c['absent']:.2%} (too few to learn)")
    for n, c in coverage.items():
        print(f"[attrs] FFHQ {n:>16}: present {c['present']:.1%} absent {c['absent']:.1%} unknown {c['unknown']:.1%}")


if __name__ == "__main__":
    main()
