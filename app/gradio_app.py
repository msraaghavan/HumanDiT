"""HumanDiT live demo: controllable face generation, attribute editing, sampler comparison, morphing, internals.

    uv run --extra demo python app/gradio_app.py
    uv run --extra demo python app/gradio_app.py --teacher runs/teacher_b/ckpt/ema.safetensors \
        --student runs/student_gan_fixed/student.safetensors --probe runs/attributes/attribute_probe.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermark import add_watermark, check_watermark, provenance_pnginfo

from humangen.data.attributes import Probe
from humangen.diffusion.guidance import GuidanceConfig, guided_model_fn
from humangen.diffusion.transport import Transport, dimension_shift
from humangen.eval.generate import load_generator, one_step_sampler
from humangen.models.layers import AttributeTokens
from humangen.rae import RAE
from humangen.utils.device import get_precision

STATE = {"any": AttributeTokens.NULL, "yes": AttributeTokens.YES, "no": AttributeTokens.NO}
# Internal Guidance scale for the teacher: the FID-10k sweep winner (1.55 -> FID-50k 3.78 vs 4.60 at the 1.78 default).
IG_TUNED = 1.55
# Teacher ODE steps: at IG 1.55, 25 steps beat 50 on FID-10k (4.79 vs 5.23) and on FID-50k (3.41 vs 3.78) at half
# the cost.
TEACHER_STEPS = 25
STUDENT_KINDS = {"gan": "GAN", "wflow": "W-Flow"}
# eval file of each student checkpoint directory, for the measured score shown next to it
STUDENT_EVALS = {
    "student_gan_fixed": "gan_fixed_1step",
    "student_wflow": "wflow_1step",
    "student_gan_anchor": "gan_anchor_1step",
    "student_gan": "gan_1step",
}
# teacher settings in the Compare tab: (steps, eval file measured at IG 1.55)
TEACHER_EVALS = (
    (10, "teacher_tuned_10step_10k"),
    (25, "teacher_tuned_25step_10k"),
    (50, "teacher_ig1.55_10k"),
)


class Engine:
    def __init__(
        self,
        teacher: str | None,
        student: str | None,
        probe: str | None,
        eval_dirs: list[str],
        onnx_dir: str | None = None,
        watermark_device: str = "auto",
    ):
        self.prec = get_precision()
        self.dev = self.prec.device
        self.transport = Transport(shift=dimension_shift(1024 * 16 * 16))
        self.wm_device = (
            ("cuda" if self.dev.type == "cuda" else "cpu") if watermark_device == "auto" else watermark_device
        )
        # Without CUDA, the 1-step mode and the decoder run through ONNX Runtime (scripts/export_onnx.py),
        # about 1.2x faster than eager PyTorch on CPU.
        self.ort_gen = self.ort_dec = None
        onnx_path = Path(onnx_dir) if onnx_dir else None
        if self.dev.type == "cpu" and onnx_path and (onnx_path / "decoder.onnx").exists():
            import onnxruntime as ort

            def session(name: str):
                return ort.InferenceSession(str(onnx_path / name), providers=["CPUExecutionProvider"])

            self.ort_dec = session("decoder.onnx")
            report = onnx_path / "export_report.json"
            exported = json.loads(report.read_text()).get("student") if report.exists() else None
            if (onnx_path / "generator_1step.onnx").exists() and student:
                if exported and Path(exported).resolve() == Path(student).resolve():
                    self.ort_gen = session("generator_1step.onnx")
                else:  # never let a stale graph of a different student answer for this one
                    print(
                        f"[demo] {onnx_path}/generator_1step.onnx was exported from {exported}, not {student}; "
                        "using PyTorch for the 1-step mode (re-run scripts/export_onnx.py)"
                    )
        self.rae = None if self.ort_dec else RAE(load_encoder=False).to(self.dev)
        self.teacher = self.student = None
        self.attr_names: list[str] = []
        self.student_name, student_eval = "student", None
        if teacher and Path(teacher).exists():
            self.teacher, self.attr_names, _ = load_generator(teacher, self.dev)
        if student and Path(student).exists():
            self.student, names, meta = load_generator(student, self.dev)
            self.attr_names = self.attr_names or names
            self.student_name = f"{STUDENT_KINDS.get(meta.get('kind', ''), '')} student".strip()
            student_eval = STUDENT_EVALS.get(Path(student).parent.name)
        if self.teacher is None and self.student is None:
            raise SystemExit("No generator checkpoint found; pass --teacher and/or --student.")
        self.fast_mode = f"1-step {self.student_name} (real-time)"
        self.teacher_mode = "Teacher (multi-step ODE)"
        self.modes = ([self.fast_mode] if self.student is not None else []) + (
            [self.teacher_mode] if self.teacher is not None else []
        )
        self.k = (self.teacher or self.student).cfg.num_attributes
        self.probe = Probe.load(probe) if probe and Path(probe).exists() else None
        self.evals, controls = {}, {}
        for d in eval_dirs:
            for f in sorted(Path(d).glob("*.json")):
                try:
                    r = json.loads(f.read_text())
                except json.JSONDecodeError:
                    continue
                if "fid_clean" in r:  # latent stats, watermark and ONNX reports are not model evals
                    self.evals[f.stem] = r
                if "controllability" in r:
                    controls[f.stem] = r["controllability"]
        self.student_score = self.score(student_eval)
        # attributes the student renders less than half the time when asked (scripts/evaluate.py --metrics control)
        ctl = controls.get(f"{student_eval}_control") or controls.get(student_eval or "") or {}
        self.student_weak = {n: v["yes"] for n, v in ctl.get("per_attribute", {}).items() if v["yes"] < 0.5}
        self.eval_dir = Path(eval_dirs[0])

    # ------------------------------------------------------------------ core
    def score(self, stem: str | None) -> str:
        r = self.evals.get(stem or "")
        return f"FID-{r['n'] // 1000}k {r['fid_clean']:.2f}" if r else "not scored"

    def noise(self, seed: int, n: int = 1) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(n, 1024, 16, 16, generator=g).to(self.dev)

    def attrs(self, choices: list[str], n: int = 1) -> torch.Tensor:
        a = torch.zeros(n, self.k, dtype=torch.long)
        for j, c in enumerate(choices[: len(self.attr_names)]):
            a[:, j] = STATE[c]
        return a.to(self.dev)

    def sync(self) -> None:
        if self.dev.type == "cuda":  # CUDA kernels are asynchronous: time only finished work
            torch.cuda.synchronize()

    @torch.no_grad()
    def sample(self, noise, attrs, mode: str, steps: int, cfg: float, ig: float, trajectory: bool = False):
        with self.prec.autocast():
            if mode == self.fast_mode and self.ort_gen is not None:
                z = self.ort_gen.run(
                    None, {"noise": noise.float().cpu().numpy(), "attrs": attrs.cpu().numpy()}
                )[0]
                return torch.from_numpy(z), []
            if mode == self.fast_mode:
                return one_step_sampler(self.student, 1)(noise, attrs).float(), []
            model = self.teacher or self.student
            fn = guided_model_fn(model, attrs, GuidanceConfig(ig_scale=ig, cfg_scale=cfg))
            out = self.transport.sample(fn, noise, int(steps), return_trajectory=trajectory)
        return (out[0].float(), out[1]) if trajectory else (out.float(), [])

    @torch.no_grad()
    def decode(self, z: torch.Tensor, mark: bool = True) -> list[Image.Image]:
        if self.ort_dec is not None:
            imgs = torch.from_numpy(self.ort_dec.run(None, {"latent": z.float().cpu().numpy()})[0])
        else:
            imgs = self.rae.decode(z)
        arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
        imgs = [Image.fromarray(a) for a in arr]
        return self.mark(imgs) if mark else imgs

    def mark(self, imgs: list[Image.Image]) -> list[Image.Image]:
        return [add_watermark(im, self.wm_device) for im in imgs]

    def probs(self, z: torch.Tensor) -> np.ndarray | None:
        return None if self.probe is None else self.probe.probs(z.mean(dim=(-2, -1))).cpu().numpy()

    def present(self, p: np.ndarray | None) -> list[str]:
        """Per face, the attributes the DINOv3 probe calls present (p > 0.5), as a caption."""
        if p is None:
            return []
        return [
            "probe: "
            + (
                ", ".join(n.replace("_", " ") for n, v in zip(self.attr_names, row, strict=True) if v > 0.5)
                or "–"
            )
            for row in p
        ]

    def weak_warning(self, mode: str, asked: list[str]) -> str:
        weak = [n for n in asked if n in self.student_weak] if mode == self.fast_mode else []
        if not weak:
            return ""
        rates = ", ".join(f"{n.replace('_', ' ')} {self.student_weak[n]:.0%}" for n in weak)
        return (
            f"⚠️ When asked, the {self.student_name} renders {rates} of the time (measured with the DINOv3 probe). "
            "Use the teacher for these attributes."
        )

    def save_png(self, img: Image.Image, details: str) -> str:
        """Write the PNG ourselves: Gradio re-encodes PIL images and would drop the provenance text chunks."""
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115
        img.save(f.name, pnginfo=provenance_pnginfo(details))
        return f.name


def seed_input(value: int, label: str = "Seed") -> gr.Number:
    with gr.Column(min_width=150):
        n = gr.Number(value=value, precision=0, label=label)
        gr.Button("🎲 Random seed", size="sm").click(lambda: random.randrange(2**31), None, n)
    return n


def build_ui(e: Engine) -> gr.Blocks:
    hw = torch.cuda.get_device_name() if e.dev.type == "cuda" else "CPU"
    with gr.Blocks(title="HumanDiT") as demo:
        gr.Markdown(
            "# HumanDiT: controllable synthetic faces\n"
            "A diffusion transformer in **DINOv3 representation space** (RAEv2), trained with **CMuon**. "
            f"**Teacher:** {TEACHER_STEPS}-step ODE with Internal Guidance {IG_TUNED} (FID-50k 3.78 at 50 steps). "
            f"**Real-time:** the one-step {e.student_name} ({e.student_score}). "
            "Every face is synthetic, invisibly watermarked (TrustMark) and saved with AI-generated PNG metadata. "
            f"Running on **{hw}**" + (" (ONNX Runtime)." if e.ort_dec is not None else ".")
        )

        def attr_block():
            radios = []
            with gr.Row():
                for name in e.attr_names:
                    radios.append(
                        gr.Radio(
                            ["any", "yes", "no"], value="any", label=name.replace("_", " "), min_width=120
                        )
                    )
            return radios

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode = gr.Radio(e.modes, value=e.modes[0], label="Sampler")
                    seed = seed_input(42)
                    count = gr.Slider(1, 8, value=4, step=1, label="Faces")
                    steps = gr.Slider(4, 100, value=TEACHER_STEPS, step=1, label="Teacher steps")
                    cfg = gr.Slider(1.0, 6.0, value=2.0, step=0.1, label="Attribute guidance (CFG, teacher)")
                    ig = gr.Slider(1.0, 3.0, value=IG_TUNED, step=0.01, label="Internal guidance (teacher)")
                    go = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Synthetic faces", columns=4, height=560)
                    timing = gr.Markdown()
                    with gr.Accordion("DINOv3 probe: P(attribute present) per face", open=False):
                        probe_out = gr.Dataframe(interactive=False, wrap=True)
            weak = ", ".join(n.replace("_", " ") for n in e.student_weak)
            gr.Markdown(
                "Attributes: *any* leaves it to the model; the one-step student ignores the CFG slider."
                + (
                    f" The {e.student_name} does not follow **{weak}** reliably (measured); use the teacher."
                    if weak
                    else ""
                )
            )
            attr_inputs = attr_block()

            def generate(mode_, seed_, count_, steps_, cfg_, ig_, *choices):
                n = int(count_)
                noise, a = e.noise(seed_, n), e.attrs(list(choices), n)
                e.sync()
                t0 = time.perf_counter()
                z, _ = e.sample(noise, a, mode_, steps_, cfg_, ig_)
                e.sync()
                t1 = time.perf_counter()
                imgs = e.decode(z, mark=False)
                e.sync()
                t2 = time.perf_counter()
                imgs = e.mark(imgs)
                t3 = time.perf_counter()
                p = e.probs(z)
                asked = {n_: c for n_, c in zip(e.attr_names, choices, strict=False) if c != "any"}
                paths = [
                    e.save_png(
                        im,
                        f"{mode_}; seed {int(seed_)} face {i}; steps {int(steps_)}; cfg {cfg_}; "
                        f"ig {ig_}; attributes {asked or 'unconditional'}",
                    )
                    for i, im in enumerate(imgs)
                ]
                msg = (
                    f"**{mode_}** · generator {1000 * (t1 - t0) / n:.0f} ms/face · decoder "
                    f"{1000 * (t2 - t1) / n:.0f} ms/face · watermark ({e.wm_device}) "
                    f"{1000 * (t3 - t2) / n:.0f} ms/face · total {t3 - t0:.2f} s for {n}"
                )
                warn = e.weak_warning(mode_, [n_ for n_, c in asked.items() if c == "yes"])
                msg += f"\n\n{warn}" if warn else ""
                table = (
                    None
                    if p is None
                    else {
                        "headers": ["face", *(a_.replace("_", " ") for a_ in e.attr_names)],
                        "data": [[i + 1, *(f"{v:.0%}" for v in row)] for i, row in enumerate(p)],
                    }
                )
                captions = e.present(p) or [None] * n
                return list(zip(paths, captions, strict=True)), msg, table

            go.click(
                generate, [mode, seed, count, steps, cfg, ig, *attr_inputs], [gallery, timing, probe_out]
            )

        with gr.Tab("Edit"):
            gr.Markdown(
                "Same seed (same noise), one attribute requested absent, left to the model, and present. "
                "Identity is *encouraged*, not guaranteed."
            )
            with gr.Row(equal_height=True):
                e_seed = seed_input(7)
                e_attr = gr.Dropdown(
                    e.attr_names, value=e.attr_names[0] if e.attr_names else None, label="Attribute"
                )
                e_mode = gr.Radio(e.modes, value=e.modes[0], label="Sampler")
                e_cfg = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="CFG (teacher)")
                e_go = gr.Button("Show absent · any · present", variant="primary")
            with gr.Row():
                e_no = gr.Image(label="absent")
                e_any = gr.Image(label="any (model's choice)")
                e_yes = gr.Image(label="present")
            e_note = gr.Markdown()

            def edit(seed_, attr_, mode_, cfg_):
                j = e.attr_names.index(attr_)
                noise = e.noise(seed_, 1).repeat(3, 1, 1, 1)
                a = torch.zeros(3, e.k, dtype=torch.long, device=e.dev)
                a[:, j] = torch.tensor(
                    [AttributeTokens.NO, AttributeTokens.NULL, AttributeTokens.YES], device=e.dev
                )
                z, _ = e.sample(noise, a, mode_, TEACHER_STEPS, cfg_, IG_TUNED)
                return *(
                    e.save_png(im, f"{mode_}; seed {int(seed_)}; {attr_} {state}; cfg {cfg_}")
                    for im, state in zip(e.decode(z), ("absent", "any", "present"), strict=True)
                ), e.weak_warning(mode_, [attr_])

            e_go.click(edit, [e_seed, e_attr, e_mode, e_cfg], [e_no, e_any, e_yes, e_note])

        with gr.Tab("Compare samplers"):
            gr.Markdown(
                "The same noise and attributes through every sampler. Times are live on this machine; the scores "
                "are measured on FFHQ-256 at IG 1.55 (lower is better; FID-10k reads higher than FID-50k, so "
                "compare like with like)."
            )
            with gr.Row(equal_height=True):
                c_seed = seed_input(11)
                c_attr = gr.Dropdown(
                    ["none", *e.attr_names], value="none", label="Request attribute (CFG 2.0)"
                )
                c_go = gr.Button("Compare", variant="primary")
            c_gallery = gr.Gallery(columns=4, height=330, label="same seed, different samplers")

            def compare(seed_, attr_):
                noise = e.noise(seed_)
                a = e.attrs([])
                if attr_ != "none":
                    a[:, e.attr_names.index(attr_)] = AttributeTokens.YES
                runs = [(e.fast_mode, 1, e.student_score)] if e.student is not None else []
                if e.teacher is not None:
                    runs += [(e.teacher_mode, s, e.score(stem)) for s, stem in TEACHER_EVALS]
                out = []
                for mode_, s, score in runs:
                    e.sync()
                    t0 = time.perf_counter()
                    z, _ = e.sample(noise, a, mode_, s, 2.0, IG_TUNED)
                    e.sync()
                    ms = 1000 * (time.perf_counter() - t0)
                    name = e.student_name if mode_ == e.fast_mode else "teacher"
                    img = e.decode(z)[0]
                    out.append(
                        (
                            e.save_png(img, f"{mode_}; {s} steps; seed {int(seed_)}; {attr_}"),
                            f"{name} · {s} step{'s' * (s > 1)} · {ms:,.0f} ms · {score}"
                            + (
                                f" · ⚠️ follows {attr_.replace('_', ' ')} {e.student_weak[attr_]:.0%} of the time"
                                if mode_ == e.fast_mode and attr_ in e.student_weak
                                else ""
                            ),
                        )
                    )
                return out

            c_go.click(compare, [c_seed, c_attr], c_gallery)

        with gr.Tab("Morph"):
            with gr.Row(equal_height=True):
                m_a = seed_input(1, "Seed A")
                m_b = seed_input(2, "Seed B")
                m_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")
                m_mode = gr.Radio(e.modes, value=e.modes[0], label="Sampler")
                m_go = gr.Button("Morph", variant="primary")
            m_video = gr.Video(label="Spherical interpolation in noise space")

            def morph(a_, b_, frames_, mode_):
                import imageio.v3 as iio

                za, zb = e.noise(a_), e.noise(b_)
                if int(a_) == int(b_):
                    b_ = int(b_) + 1  # identical seeds would make the slerp angle 0 (division by zero)
                    zb = e.noise(b_)
                cos = ((za * zb).sum() / (za.norm() * zb.norm())).clamp(-1 + 1e-6, 1 - 1e-6)
                omega = torch.arccos(cos)
                ts = torch.linspace(0, 1, int(frames_), device=e.dev)
                frames = []
                for i in range(0, len(ts), 8):
                    t = ts[i : i + 8].view(-1, 1, 1, 1)
                    z = (torch.sin((1 - t) * omega) * za + torch.sin(t * omega) * zb) / torch.sin(omega)
                    lat, _ = e.sample(z, e.attrs([], len(t)), mode_, TEACHER_STEPS, 1.0, IG_TUNED)
                    frames += [np.asarray(im) for im in e.decode(lat)]
                path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name  # noqa: SIM115
                iio.imwrite(path, np.stack(frames + frames[::-1]), fps=12, codec="libx264")
                return path

            m_go.click(morph, [m_a, m_b, m_frames, m_mode], m_video)

        with gr.Tab("Under the hood"):
            gr.Markdown(
                "Teacher denoising trajectory: the clean-latent prediction x̂₀ decoded at several ODE steps."
            )
            with gr.Row(equal_height=True):
                u_seed = seed_input(3)
                u_go = gr.Button("Show trajectory", variant="primary")
            u_gallery = gr.Gallery(columns=6, height=260)
            rows = sorted(
                (
                    [
                        stem,
                        r.get("steps"),
                        f"{r['n']:,}",
                        r["fid_clean"],
                        r.get("fd_dinov2"),
                        r.get("precision"),
                        r.get("recall"),
                    ]
                    for stem, r in e.evals.items()
                ),
                key=lambda row: row[3],
            )
            gr.Dataframe(
                value={
                    "headers": ["run", "steps", "samples", "FID (clean)", "FD_DINOv2", "precision", "recall"],
                    "data": rows,
                }
                if rows
                else None,
                label="Measured metrics (runs/eval*; scripts/evaluate.py)",
                interactive=False,
            )
            # the teacher's audit (the alphabetically first file would be the collapsed GAN's)
            pairs = [
                p
                for n in ("teacher_tuned", "teacher")
                if (p := e.eval_dir / f"{n}_nearest_pairs.png").exists()
            ]
            if pairs:
                gr.Image(
                    str(pairs[0]),
                    label="Memorisation audit: most similar generated (top) vs training (bottom)",
                )

            def trajectory(seed_):
                if e.teacher is None:
                    return []
                _, traj = e.sample(
                    e.noise(seed_), e.attrs([]), e.teacher_mode, TEACHER_STEPS, 1.0, IG_TUNED, trajectory=True
                )
                pick = sorted({round(x) for x in np.linspace(0, TEACHER_STEPS - 1, 6)})
                imgs = e.decode(torch.cat([traj[i] for i in pick]).float())
                return [(img, f"step {i + 1}/{TEACHER_STEPS}") for i, img in zip(pick, imgs, strict=True)]

            u_go.click(trajectory, u_seed, u_gallery)

        with gr.Tab("Check watermark"):
            w_in = gr.Image(type="pil", label="Upload an image")
            w_out = gr.Markdown()
            w_in.change(
                lambda im: (
                    ""
                    if im is None
                    else (
                        "✅ HumanDiT watermark detected"
                        if check_watermark(im, e.wm_device)
                        else "No HumanDiT watermark found"
                    )
                ),
                w_in,
                w_out,
            )
        gr.Markdown(
            "Research demo. Trained on FFHQ (CC BY-NC-SA 4.0) with RAEv2 (CC BY-NC 4.0) and DINOv3 (DINOv3 "
            "License) components: **non-commercial use only**. See MODEL_CARD.md for limitations and bias."
        )
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--student", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--eval-dirs", nargs="+", default=["runs/eval", "runs/eval_extra", "runs/eval_local"])
    ap.add_argument("--onnx-dir", default="onnx", help="ONNX graphs used when no CUDA GPU is available")
    ap.add_argument("--watermark-device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    engine = Engine(
        args.teacher, args.student, args.probe, args.eval_dirs, args.onnx_dir, args.watermark_device
    )
    build_ui(engine).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
