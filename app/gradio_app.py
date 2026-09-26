"""HumanDiT live demo: controllable face generation, attribute editing, morphing, and model internals.

    uv run --extra demo python app/gradio_app.py
    uv run --extra demo python app/gradio_app.py --teacher runs/teacher_b/ckpt/ema.safetensors \
        --student runs/student_wflow/student.safetensors --probe cache/attribute_probe.pt
"""

from __future__ import annotations

import argparse
import json
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


class Engine:
    def __init__(self, teacher: str | None, student: str | None, probe: str | None, eval_dir: str,
                 onnx_dir: str | None = None):
        self.prec = get_precision()
        self.dev = self.prec.device
        self.transport = Transport(shift=dimension_shift(1024 * 16 * 16))
        # Without CUDA, the 1-step mode and the decoder run through ONNX Runtime (scripts/export_onnx.py),
        # about 1.7x faster than eager PyTorch on CPU.
        self.ort_gen = self.ort_dec = None
        onnx_path = Path(onnx_dir) if onnx_dir else None
        if self.dev.type == "cpu" and onnx_path and (onnx_path / "decoder.onnx").exists():
            import onnxruntime as ort

            def session(name: str):
                return ort.InferenceSession(str(onnx_path / name), providers=["CPUExecutionProvider"])

            self.ort_dec = session("decoder.onnx")
            if (onnx_path / "generator_1step.onnx").exists():
                self.ort_gen = session("generator_1step.onnx")
        self.rae = None if self.ort_dec else RAE(load_encoder=False).to(self.dev)
        self.teacher = self.student = None
        self.attr_names: list[str] = []
        if teacher and Path(teacher).exists():
            self.teacher, self.attr_names, self.teacher_meta = load_generator(teacher, self.dev)
        if student and Path(student).exists():
            self.student, names, self.student_meta = load_generator(student, self.dev)
            self.attr_names = self.attr_names or names
        if self.teacher is None and self.student is None:
            raise SystemExit("No generator checkpoint found; pass --teacher and/or --student.")
        self.k = (self.teacher or self.student).cfg.num_attributes
        self.probe = Probe.load(probe) if probe and Path(probe).exists() else None
        self.eval_dir = Path(eval_dir)

    # ------------------------------------------------------------------ core
    def noise(self, seed: int, n: int = 1) -> torch.Tensor:
        g = torch.Generator().manual_seed(int(seed))
        return torch.randn(n, 1024, 16, 16, generator=g).to(self.dev)

    def attrs(self, choices: list[str], n: int = 1) -> torch.Tensor:
        a = torch.zeros(n, self.k, dtype=torch.long)
        for j, c in enumerate(choices[: len(self.attr_names)]):
            a[:, j] = STATE[c]
        return a.to(self.dev)

    @torch.no_grad()
    def sample(self, noise, attrs, mode: str, steps: int, cfg: float, ig: float, trajectory: bool = False):
        with self.prec.autocast():
            if mode.startswith("1-step") and self.ort_gen is not None:
                z = self.ort_gen.run(None, {"noise": noise.float().cpu().numpy(), "attrs": attrs.cpu().numpy()})[0]
                return torch.from_numpy(z), []
            if mode.startswith("1-step") and self.student is not None:
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
        return [add_watermark(im) for im in imgs] if mark else imgs

    def describe(self, z: torch.Tensor) -> list[dict]:
        if self.probe is None:
            return []
        p = self.probe.probs(z.mean(dim=(-2, -1))).cpu().numpy()
        return [{n: round(float(v), 3) for n, v in zip(self.probe.display_names, row, strict=True)} for row in p]

    def save_png(self, img: Image.Image, details: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115
        img.save(f.name, pnginfo=provenance_pnginfo(details))
        return f.name


def build_ui(e: Engine) -> gr.Blocks:
    modes = (["1-step (real-time student)"] if e.student is not None else []) + (
        ["Teacher (multi-step ODE)"] if e.teacher is not None else [])
    with gr.Blocks(title="HumanDiT") as demo:
        gr.Markdown(
            "# HumanDiT: controllable synthetic faces\n"
            "Diffusion transformer in **DINOv3 representation space** (RAEv2), trained with **CMuon**, "
            "with a **one-step** student. Every face is synthetic and invisibly watermarked. "
            f"Running on **{torch.cuda.get_device_name() if e.dev.type == 'cuda' else 'CPU'}**"
            + (" (ONNX Runtime)." if e.ort_dec is not None else ".")
        )
        attr_inputs = []

        def attr_block():
            radios = []
            with gr.Row():
                for name in e.attr_names:
                    radios.append(gr.Radio(["any", "yes", "no"], value="any", label=name.replace("_", " "), min_width=120))
            return radios

        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    mode = gr.Radio(modes, value=modes[0], label="Sampler")
                    seed = gr.Number(value=42, precision=0, label="Seed")
                    count = gr.Slider(1, 8, value=4, step=1, label="Faces")
                    steps = gr.Slider(4, 100, value=50, step=1, label="Teacher steps")
                    cfg = gr.Slider(1.0, 6.0, value=2.5, step=0.1, label="Attribute guidance (CFG)")
                    ig = gr.Slider(1.0, 3.0, value=1.78, step=0.02, label="Internal guidance")
                    go = gr.Button("Generate", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Synthetic faces", columns=4, height=560, format="png")
                    timing = gr.Markdown()
                    probe_out = gr.JSON(label="DINOv3 probe: P(attribute present) per face")
            attr_inputs = attr_block()

            def generate(mode_, seed_, count_, steps_, cfg_, ig_, *choices):
                n = int(count_)
                noise, a = e.noise(seed_, n), e.attrs(list(choices), n)
                if e.dev.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                z, _ = e.sample(noise, a, mode_, steps_, cfg_, ig_)
                t1 = time.perf_counter()
                imgs = e.decode(z, mark=False)
                t2 = time.perf_counter()
                imgs = [add_watermark(im) for im in imgs]
                t3 = time.perf_counter()
                msg = (f"**{mode_}** · generator {1000 * (t1 - t0) / n:.0f} ms/face · decoder "
                       f"{1000 * (t2 - t1) / n:.0f} ms/face · watermark (CPU) {1000 * (t3 - t2) / n:.0f} ms/face · "
                       f"total {t3 - t0:.2f} s for {n}")
                return imgs, msg, e.describe(z)

            go.click(generate, [mode, seed, count, steps, cfg, ig, *attr_inputs], [gallery, timing, probe_out])

        with gr.Tab("Edit"):
            gr.Markdown("Same seed (same noise), one attribute flipped. Identity is *encouraged*, not guaranteed.")
            with gr.Row():
                e_seed = gr.Number(value=7, precision=0, label="Seed")
                e_attr = gr.Dropdown(e.attr_names, value=e.attr_names[0] if e.attr_names else None, label="Attribute")
                e_mode = gr.Radio(modes, value=modes[-1], label="Sampler")
                e_go = gr.Button("Show absent vs present", variant="primary")
            with gr.Row():
                e_no, e_yes = gr.Image(label="absent", format="png"), gr.Image(label="present", format="png")

            def edit(seed_, attr_, mode_):
                j = e.attr_names.index(attr_)
                noise = e.noise(seed_, 2)[:1].repeat(2, 1, 1, 1)
                a = torch.zeros(2, e.k, dtype=torch.long, device=e.dev)
                a[0, j], a[1, j] = AttributeTokens.NO, AttributeTokens.YES
                z, _ = e.sample(noise, a, mode_, 50, 3.0, 1.78)
                no, yes = e.decode(z)
                return no, yes

            e_go.click(edit, [e_seed, e_attr, e_mode], [e_no, e_yes])

        with gr.Tab("Morph"):
            with gr.Row():
                m_a = gr.Number(value=1, precision=0, label="Seed A")
                m_b = gr.Number(value=2, precision=0, label="Seed B")
                m_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")
                m_mode = gr.Radio(modes, value=modes[0], label="Sampler")
                m_go = gr.Button("Morph", variant="primary")
            m_video = gr.Video(label="Spherical interpolation in noise space")

            def morph(a_, b_, frames_, mode_):
                import imageio.v3 as iio

                za, zb = e.noise(a_), e.noise(b_)
                omega = torch.arccos((za * zb).sum() / (za.norm() * zb.norm()))
                ts = torch.linspace(0, 1, int(frames_), device=e.dev)
                frames = []
                for i in range(0, len(ts), 8):
                    t = ts[i : i + 8].view(-1, 1, 1, 1)
                    z = (torch.sin((1 - t) * omega) * za + torch.sin(t * omega) * zb) / torch.sin(omega)
                    lat, _ = e.sample(z, e.attrs([], len(t)), mode_, 30, 1.0, 1.78)
                    frames += [np.asarray(im) for im in e.decode(lat)]
                path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name  # noqa: SIM115
                iio.imwrite(path, np.stack(frames + frames[::-1]), fps=12, codec="libx264")
                return path

            m_go.click(morph, [m_a, m_b, m_frames, m_mode], m_video)

        with gr.Tab("Under the hood"):
            gr.Markdown("Teacher denoising trajectory: the clean-latent prediction x̂₀ decoded at several ODE steps.")
            with gr.Row():
                u_seed = gr.Number(value=3, precision=0, label="Seed")
                u_go = gr.Button("Show trajectory", variant="primary")
            u_gallery = gr.Gallery(columns=6, height=260, format="png")
            results = {}
            for f in sorted(e.eval_dir.glob("*.json")):
                try:
                    r = json.loads(f.read_text())
                    results[r.get("tag", f.stem)] = {k: r.get(k) for k in (
                        "fid_clean", "fid_legacy_tensorflow", "fd_dinov2", "precision", "recall", "steps")}
                except Exception:
                    continue
            gr.JSON(value=results or {"note": "run scripts/evaluate.py to populate"}, label="Measured metrics (runs/eval)")
            pairs = sorted(e.eval_dir.glob("*_nearest_pairs.png"))
            if pairs:
                gr.Image(str(pairs[0]), label="Memorisation audit: most similar generated (top) vs training (bottom)")

            def trajectory(seed_):
                if e.teacher is None:
                    return []
                _, traj = e.sample(e.noise(seed_), e.attrs([]), "Teacher", 50, 1.0, 1.78, trajectory=True)
                pick = [0, 5, 10, 20, 35, 49]
                return [(img, f"step {i + 1}") for i, img in zip(pick, e.decode(torch.cat([traj[i] for i in pick]).float()), strict=True)]

            u_go.click(trajectory, u_seed, u_gallery)

        with gr.Tab("Check watermark"):
            w_in = gr.Image(type="pil", label="Upload an image")
            w_out = gr.Markdown()
            w_in.change(lambda im: "" if im is None else ("✅ HumanDiT watermark detected" if check_watermark(im)
                                                           else "No HumanDiT watermark found"), w_in, w_out)
        gr.Markdown("Research demo. Trained on FFHQ (CC BY-NC-SA 4.0) with RAEv2 (CC BY-NC 4.0) and DINOv3 (DINOv3 "
                    "License) components: **non-commercial use only**. See MODEL_CARD.md for limitations and bias.")
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--student", default="runs/student_wflow/student.safetensors")
    ap.add_argument("--probe", default="cache/attribute_probe.pt")
    ap.add_argument("--eval-dir", default="runs/eval")
    ap.add_argument("--onnx-dir", default="onnx", help="ONNX graphs used when no CUDA GPU is available")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    engine = Engine(args.teacher, args.student, args.probe, args.eval_dir, args.onnx_dir)
    build_ui(engine).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
