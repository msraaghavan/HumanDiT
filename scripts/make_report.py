"""Build the README results section from measured JSON files (no hand-typed numbers).

    uv run python scripts/make_report.py --runs runs
Replaces the text between <!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def _fmt(v, nd=2) -> str:
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def build(runs: Path) -> str:
    lines = []
    ev = {p.stem: _load(p) for p in sorted((runs / "eval").glob("*.json"))}
    t0 = ev.get("teacher") or {}
    rows = [("teacher", f"Teacher, {t0.get('steps', '?')}-step ODE + Internal Guidance"),
            ("wflow_1step", "W-Flow student, 1 step"), ("gan_1step", "GAN (adversarial post-training) student, 1 step"),
            ("legacy_dcgan_64px", "*Before:* original DCGAN (bugs fixed), 64×64")]
    if any(k in ev for k, _ in rows):
        n = next((ev[k]["n"] for k, _ in rows if k in ev), "?")
        lines += [f"### Generation quality (FFHQ 256×256, {n:,} samples per model)", "",
                  "| Model | NFE | FID (clean) ↓ | FID (legacy TF) ↓ | FD_DINOv2 ↓ | Precision ↑ | Recall ↑ | img/s (4090, bs16) |",
                  "|---|---|---|---|---|---|---|---|"]
        for key, label in rows:
            r = ev.get(key)
            if not r:
                continue
            sp = (r.get("speed") or {}).get("batch16", {}).get("images_per_s")
            lines.append(f"| {label} | {r.get('steps', 1)} | {_fmt(r.get('fid_clean'))} | {_fmt(r.get('fid_legacy_tensorflow'))} | "
                         f"{_fmt(r.get('fd_dinov2'), 1)} | {_fmt(r.get('precision'), 3)} | {_fmt(r.get('recall'), 3)} | {_fmt(sp, 1)} |")
        lines.append("")
        t = ev.get("teacher") or {}
        if t.get("controllability"):
            c = t["controllability"]
            lines += [f"### Attribute control (teacher, CFG {c['cfg_scale']}; scored by the DINOv3 probe on generated latents)", "",
                      "| Attribute | asked *present* → probe agrees | asked *absent* → probe agrees | base rate (unconditional) |",
                      "|---|---|---|---|"]
            base = c.get("unconditional_positive_rate") or {}
            for name, v in c["per_attribute"].items():
                lines.append(f"| {name.replace('_', ' ')} | {v['yes']:.1%} | {v['no']:.1%} | {_fmt(base.get(name), 3)} |")
            lines += ["", f"Mean controllability accuracy: **{c['mean_accuracy']:.1%}**.", ""]
        if t.get("memorization"):
            m = t["memorization"]
            lines += [f"### Memorisation audit (DINOv3 space, {min(t['n'], 10000):,} samples vs every training face)", "",
                      f"- Nearest-training-face cosine similarity: generated median **{m['gen_nn_sim_p50']}** "
                      f"(p99 {m['gen_nn_sim_p99']}) vs training-to-training median {m['train_nn_sim_p50']} (p99 {m['train_nn_sim_p99']}).",
                      f"- Fraction of samples closer to a training face than 99.9% of real faces are to each other: "
                      f"**{m['frac_gen_above_threshold']:.3%}**.", ""]
    gate = _load(runs / "rae_gate" / "metrics.json")
    if gate:
        lines += ["### RAEv2 decoder on faces (reconstruction gate)", "",
                  f"PSNR **{gate['psnr_db']} dB**, LPIPS **{gate['lpips_alex']}**"
                  + (f", rFID **{gate['rfid_clean']}**" if "rfid_clean" in gate else "") + f" on {gate['n']} FFHQ images.", ""]
    attrs = _load(runs / "attributes" / "attributes.json")
    if attrs:
        m = attrs["celeba_test_metrics_all"]
        lines += ["### DINOv3 attribute probe (CelebA official test split)", "",
                  "| Attribute | Balanced acc. | AUROC | FFHQ labelled present / absent / abstain |", "|---|---|---|---|"]
        for name, col in zip(attrs["names"], attrs["celeba_columns"], strict=True):
            cov = attrs["ffhq_label_coverage"][name]
            lines.append(f"| {name.replace('_', ' ')} | {m[col]['balanced_acc']:.3f} | {m[col]['auroc']:.3f} | "
                         f"{cov['present']:.0%} / {cov['absent']:.0%} / {cov['unknown']:.0%} |")
        lines.append("")
    ts = _load(runs / "teacher_b" / "train_summary.json")
    if ts:
        lines += ["### Training cost", "",
                  f"Teacher: {ts['steps']:,} steps, {ts['samples_seen'] / 1e6:.2f}M samples, {ts['hours']} h on "
                  f"{ts['gpu']} ({ts['params_m']}M params, {ts['precision']}).", ""]
    return "\n".join(lines) if lines else "_Results will appear here after the RunPod run._"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # tables contain arrows; Windows consoles default to cp1252
    body = build(Path(args.runs))
    readme = Path(args.readme) if args.readme else None
    text = readme.read_text(encoding="utf-8") if readme and readme.is_file() else ""
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if readme and start in text and end in text:
        text = text[: text.index(start) + len(start)] + "\n" + body + "\n" + text[text.index(end):]
        readme.write_text(text, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
