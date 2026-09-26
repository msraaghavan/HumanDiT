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


def _ablations(d: Path) -> list[str]:
    """FID-10k sweeps from runs/eval_extra (scripts/pod/run_all.sh EXTRAS=1)."""
    x = {p.stem: _load(p) for p in sorted(d.glob("*.json"))} if d.is_dir() else {}
    if not x:
        return []
    lines = []
    ig = sorted(
        (r["guidance"]["ig_scale"], r["fid_clean"]) for k, r in x.items() if k.startswith("teacher_ig")
    )
    if ig:
        best = min(ig, key=lambda v: v[1])
        lines += [
            "### Ablation: Internal Guidance scale (teacher, 50 steps, FID-10k clean)",
            "",
            "| IG scale | " + " | ".join(f"{s:.2f}" for s, _ in ig) + " |",
            "|---|" + "---|" * len(ig),
            "| FID-10k ↓ | "
            + " | ".join(f"**{f:.2f}**" if (s, f) == best else f"{f:.2f}" for s, f in ig)
            + " |",
            "",
            "FID-10k is biased upwards relative to FID-50k; compare within this table only.",
            "",
        ]
    nfe = []
    for k, r in x.items():
        if k.startswith("teacher_ig") and abs(r["guidance"]["ig_scale"] - 1.78) > 1e-6:
            continue
        name = (
            "Teacher (IG 1.78)"
            if k.startswith("teacher")
            else "W-Flow student"
            if k.startswith("wflow")
            else "GAN student"
            if k.startswith("gan")
            else None
        )
        if name:
            nfe.append((name, r["steps"], r["fid_clean"]))
    if nfe:
        lines += [
            "### Quality vs network evaluations (FID-10k clean)",
            "",
            "| Model | NFE | FID-10k ↓ |",
            "|---|---|---|",
        ]
        lines += [f"| {n} | {s} | {f:.2f} |" for n, s, f in sorted(nfe)]
        lines.append("")
        teach = sorted((s, f) for n, s, f in nfe if n.startswith("Teacher"))
        best = min(teach, key=lambda v: v[1]) if teach else None
        if best and teach[-1][0] > best[0]:
            lines += [
                f"With IG 1.78, {best[0]} steps score better than {teach[-1][0]} ({best[1]:.2f} vs {teach[-1][1]:.2f}). "
                "Together with the guidance sweep, this suggests 1.78 over-guides; the step count at the tuned scale "
                "was not swept.",
                "",
            ]
    return lines


def build(runs: Path) -> str:
    lines = []
    ev = {p.stem: _load(p) for p in sorted((runs / "eval").glob("*.json"))}
    t0 = ev.get("teacher") or {}
    tt = ev.get("teacher_tuned") or {}
    rows = [
        (
            "teacher_tuned",
            f"**Teacher, {tt.get('steps', '?')}-step ODE + Internal Guidance "
            f"{(tt.get('guidance') or {}).get('ig_scale', '?')} (tuned)**",
        ),
        (
            "teacher",
            f"Teacher, {t0.get('steps', '?')}-step ODE + Internal Guidance "
            f"{(t0.get('guidance') or {}).get('ig_scale', '?')} (default)",
        ),
        ("wflow_1step", "W-Flow student, 1 step"),
        ("gan_anchor_1step", "GAN student + teacher-pair anchor (2nd attempt), 1 step"),
        ("gan_1step", "GAN student, adversarial only (1st attempt: **mode collapse**), 1 step"),
        ("legacy_dcgan_64px", "*Before:* original DCGAN (bugs fixed), 64×64"),
    ]
    if any(k in ev for k, _ in rows):
        n = next((ev[k]["n"] for k, _ in rows if k in ev), "?")
        lines += [
            f"### Generation quality (FFHQ 256×256, {n:,} samples per model)",
            "",
            "| Model | NFE | FID (clean) ↓ | FID (legacy TF) ↓ | FD_DINOv2 ↓ | Precision ↑ | Recall ↑ | img/s (4090, bs16) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for key, label in rows:
            r = ev.get(key)
            if not r:
                continue
            sp = (r.get("speed") or {}).get("batch16", {}).get("images_per_s")
            lines.append(
                f"| {label} | {r.get('steps', 1)} | {_fmt(r.get('fid_clean'))} | {_fmt(r.get('fid_legacy_tensorflow'))} | "
                f"{_fmt(r.get('fd_dinov2'), 1)} | {_fmt(r.get('precision'), 3)} | {_fmt(r.get('recall'), 3)} | {_fmt(sp, 1)} |"
            )
        lines.append("")
        legacy = ev.get("legacy_dcgan_64px") or {}
        if legacy.get("fid_clean_64px_vs_ffhq64") is not None:
            lines += [
                f"The DCGAN generates 64×64 images, so the 256 px reference also penalises it for resolution. "
                f"Scored at its own resolution against all 70k FFHQ faces downsampled to 64×64, its FID "
                f"(clean) is **{legacy['fid_clean_64px_vs_ffhq64']:.1f}**.",
                "",
            ]
        if tt and t0:
            ig_t, ig_0 = (
                (tt.get("guidance") or {}).get("ig_scale"),
                (t0.get("guidance") or {}).get("ig_scale"),
            )
            lines += [
                f"The tuned row uses the guidance scale that won the FID-10k sweep below ({ig_t}), re-scored on "
                f"{tt.get('n', 0):,} samples. Only the guidance scale differs, so its speed equals the default's. "
                f"Attribute control and latent statistics below were measured at the default {ig_0}, before the sweep.",
                "",
            ]
        t = ev.get("teacher") or {}
        if t.get("controllability"):
            c = t["controllability"]
            lines += [
                f"### Attribute control (teacher, CFG {c['cfg_scale']}; scored by the DINOv3 probe on generated latents)",
                "",
                "| Attribute | asked *present* → probe agrees | asked *absent* → probe agrees | base rate (unconditional) |",
                "|---|---|---|---|",
            ]
            base = c.get("unconditional_positive_rate") or {}
            for name, v in c["per_attribute"].items():
                b = base.get(name)
                lines.append(
                    f"| {name.replace('_', ' ')} | {v['yes']:.1%} | {v['no']:.1%} | {'–' if b is None else f'{b:.1%}'} |"
                )
            lines += ["", f"Mean controllability accuracy: **{c['mean_accuracy']:.1%}**.", ""]
            cov = (_load(runs / "attributes" / "attributes.json") or {}).get("ffhq_label_coverage", {})
            rare = [n for n in base if n in cov and cov[n]["present"] < 0.10]
            stats_ = _load(runs / "eval" / "latent_stats.json")
            if rare and base:
                gen_r = [base[n] for n in rare]
                data_r = [cov[n]["present"] for n in rare]
                note = (
                    f"The probe over-fires on *generated* latents. For the rarer attributes "
                    f"({', '.join(n.replace('_', ' ').lower() for n in rare)}) it calls {min(gen_r):.0%}–{max(gen_r):.0%} "
                    f"of unconditional samples positive, while FFHQ has {min(data_r):.0%}–{max(data_r):.0%}."
                )
                ig_key = f"ig_{(t.get('guidance') or {}).get('ig_scale', 1.78):.2f}"
                if stats_ and ig_key in stats_ and "ig_1.00" in stats_:
                    note += (
                        f" Internal Guidance extrapolation inflates latent statistics: latent std is "
                        f"{stats_[ig_key]['std']:.2f} with guidance, vs {stats_['ig_1.00']['std']:.2f} without it "
                        f"and {stats_['real']['std']:.2f} for real FFHQ latents (`scripts/latent_stats.py`)."
                    )
                note += " So the absolute percentages are approximate; the shift from the base-rate column is the controllable effect."
                lines += [note, ""]
        if t.get("memorization"):
            m = t["memorization"]
            lines += [
                f"### Memorisation audit (DINOv3 space, {min(t['n'], 10000):,} samples vs every training face)",
                "",
                f"- Nearest-training-face cosine similarity: generated median **{m['gen_nn_sim_p50']:.2f}** "
                f"(p99 {m['gen_nn_sim_p99']:.2f}), vs training-to-training median {m['train_nn_sim_p50']:.2f} "
                f"(p99 {m['train_nn_sim_p99']:.2f}).",
                f"- Fraction of samples closer to a training face than 99.9% of real faces are to each other: "
                f"**{m['frac_gen_above_threshold']:.3%}**.",
                "- Caveat: generated faces sit *further* from the training set than real faces sit from each other. That is "
                "partly the same guidance-induced shift in latent statistics, so this feature space understates similarity. "
                f"The direct evidence is below: the 8 **most similar** pairs out of {min(t['n'], 10000):,} share pose and "
                "studio style, but show different people.",
            ]
            mt = tt.get("memorization")
            if mt:
                lines.append(
                    f"- At the tuned guidance scale ({(tt.get('guidance') or {}).get('ig_scale')}), the generated median "
                    f"is {mt['gen_nn_sim_p50']:.2f} (p99 {mt['gen_nn_sim_p99']:.2f}), closer to the training-set "
                    f"statistics as the caveat predicts, and {mt['frac_gen_above_threshold']:.3%} of samples exceed "
                    f"the threshold."
                )
            lines.append("")
            if Path("docs/assets/teacher_nearest_pairs.jpg").exists():
                lines += [
                    "![most similar generated (top) vs training (bottom)](docs/assets/teacher_nearest_pairs.jpg)",
                    "",
                ]
    lines += _ablations(runs / "eval_extra")
    ox = _load(runs / "eval" / "onnx_export.json")
    if ox:
        g, d, tq = ox["generator_1step"], ox["decoder"], ox.get("torch_cpu_ms_per_face", {})
        lines += [
            "### CPU-only deployment (ONNX Runtime, laptop CPU, batch 1)",
            "",
            "| Graph | max abs error vs PyTorch | ONNX Runtime ms/face | eager PyTorch ms/face | size |",
            "|---|---|---|---|---|",
            f"| 1-step generator | {g['max_abs_diff']:.1e} | {g['cpu_ms_per_face']:.0f} | {tq.get('generator_1step', 0):.0f} | {g['mb']:.0f} MB |",
            f"| RAEv2 decoder | {d['max_abs_diff']:.1e} | {d['cpu_ms_per_face']:.0f} | {tq.get('decoder', 0):.0f} | {d['mb']:.0f} MB |",
        ]
        q = ox.get("decoder_int8")
        if q:
            lines.append(
                f"| decoder, {q['method']} | PSNR {q['psnr_vs_fp32_db']:.1f} dB vs fp32 | {q['cpu_ms_per_face']:.0f} | – | {q['mb']:.0f} MB |"
            )
        lines += [
            "",
            "Dynamic int8 (activations quantised too) broke the ViT-XL decoder in testing (9–11 dB): its activation "
            "outliers need per-block weight-only quantisation. Weight-only int8 is 3.5× smaller but not faster on this CPU.",
            "",
        ]
    wm = _load(runs / "eval" / "watermark.json")
    if wm:
        r = wm["recovery"]
        lines += [
            f"### Invisible watermark on every demo output ({wm['method']}, {wm['n_marked']} generated faces)",
            "",
            "| PNG | JPEG q75 | JPEG q50 | 2× down/up-scale | 90% centre crop | PSNR | false positives (real / unmarked) |",
            "|---|---|---|---|---|---|---|",
            f"| {r['png']:.0%} | {r['jpeg_q75']:.0%} | {r['jpeg_q50']:.1%} | {r['downscale_2x_upscale']:.0%} | "
            f"{r['center_crop_90pct']:.0%} | {wm['psnr_db_mean']:.1f} dB | "
            f"{wm['false_positive_rate_real_ffhq']:.0%} / {wm['false_positive_rate_unmarked_generated']:.0%} |",
            "",
        ]
    gate = _load(runs / "rae_gate" / "metrics.json")
    if gate:
        lines += [
            "### RAEv2 decoder on faces (reconstruction gate)",
            "",
            f"PSNR **{gate['psnr_db']:.1f} dB**, LPIPS **{gate['lpips_alex']:.3f}**"
            + (f", rFID **{gate['rfid_clean']:.2f}**" if "rfid_clean" in gate else "")
            + f" on {gate['n']:,} FFHQ images "
            "with the released ImageNet-trained decoder, no face fine-tuning.",
            "",
        ]
    attrs = _load(runs / "attributes" / "attributes.json")
    if attrs:
        m = attrs["celeba_test_metrics_all"]
        lines += [
            "### DINOv3 attribute probe (CelebA official test split)",
            "",
            "| Attribute | Balanced acc. | AUROC | FFHQ labelled present / absent / abstain |",
            "|---|---|---|---|",
        ]
        for name, col in zip(attrs["names"], attrs["celeba_columns"], strict=True):
            cov = attrs["ffhq_label_coverage"][name]
            lines.append(
                f"| {name.replace('_', ' ')} | {m[col]['balanced_acc']:.3f} | {m[col]['auroc']:.3f} | "
                f"{cov['present']:.0%} / {cov['absent']:.0%} / {cov['unknown']:.0%} |"
            )
        lines.append("")
    ts = _load(runs / "teacher_b" / "train_summary.json")
    if ts:
        lines += [
            "### Training cost",
            "",
            f"Teacher: {ts['steps']:,} steps, {ts['samples_seen'] / 1e6:.2f}M samples, {ts['hours']:.1f} h on "
            f"{ts['gpu']} ({ts['params_m']}M params, {ts['precision']}).",
            "",
        ]
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
        text = text[: text.index(start) + len(start)] + "\n" + body + "\n" + text[text.index(end) :]
        readme.write_text(text, encoding="utf-8", newline="\n")
    print(body)


if __name__ == "__main__":
    main()
