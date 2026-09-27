"""HumanDiT live demo, written for anyone: make faces, change one feature, blend two faces, watch it draw, compare
instant vs best quality, check the watermark. Everything technical lives in one "For engineers" tab.

    uv run --extra demo python app/gradio_app.py
    uv run --extra demo python app/gradio_app.py --teacher runs/teacher_b/ckpt/ema.safetensors \
        --student runs/student_gan_fixed/student.safetensors --probe runs/attributes/attribute_probe.pt
"""

from __future__ import annotations

import argparse
import base64
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
# teacher settings in the Compare tab: (steps, eval files measured at IG 1.55, the FID-50k one first)
TEACHER_EVALS = (
    (10, ("teacher_tuned_10step_10k",)),
    (25, ("teacher_tuned_25step", "teacher_tuned_25step_10k")),
    (50, ("teacher_tuned", "teacher_ig1.55_10k")),
)

# The header reel (docs/assets/showreel.mp4) carries a soundtrack; browsers only autoplay muted video, and gr.Video has
# no `muted` option, so mute it as it appears (its controls can unmute it) and keep it playing.
REEL_MUTED = """<script>
new MutationObserver(() => document.querySelectorAll("#reel video").forEach(v => {
  if (!v.dataset.muted) { v.dataset.muted = 1; v.muted = true; v.play().catch(() => {}); }
})).observe(document.documentElement, { childList: true, subtree: true });
// browsers may pause muted video while the page is hidden: resume the tour when it is shown again
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) document.querySelectorAll("#reel video").forEach(v => v.play().catch(() => {}));
});
</script>"""


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
        self.student_eval = student_eval
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
        imgs = []
        # The ViT-XL decoder's activations for 8 faces at once overflow a 4 GB GPU into shared system memory
        # (about 4 s a face instead of 0.16 s on the GTX 1650 Ti), so decode a few at a time.
        for zc in z.split(DECODE_CHUNK):
            if self.ort_dec is not None:
                x = torch.from_numpy(self.ort_dec.run(None, {"latent": zc.float().cpu().numpy()})[0])
            else:
                x = self.rae.decode(zc)
            arr = (x.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            imgs += [Image.fromarray(a) for a in arr]
        return self.mark(imgs) if mark else imgs

    def warm_up(self) -> None:
        """Run every path once before the page opens (CUDA kernels, cuDNN autotuning, TrustMark), so the first
        visitor's timings are the steady-state ones rather than one-off start-up costs."""
        t0 = time.perf_counter()
        noise, a = self.noise(0, DECODE_CHUNK), self.attrs([], DECODE_CHUNK)
        for mode in self.modes:
            z, _ = self.sample(noise, a, mode, 2, 2.0, IG_TUNED)
            self.decode(z)
            self.probs(z)
        self.sync()
        print(f"[demo] warmed up in {time.perf_counter() - t0:.1f} s")

    def mark(self, imgs: list[Image.Image]) -> list[Image.Image]:
        return [add_watermark(im, self.wm_device) for im in imgs]

    def probs(self, z: torch.Tensor) -> np.ndarray | None:
        return None if self.probe is None else self.probe.probs(z.mean(dim=(-2, -1))).cpu().numpy()

    def save_png(self, img: Image.Image, details: str) -> str:
        """Write the PNG ourselves: Gradio re-encodes PIL images and would drop the provenance text chunks."""
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)  # noqa: SIM115
        img.save(f.name, pnginfo=provenance_pnginfo(details))
        return f.name


ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
# plain-English names for the 12 attributes (the model's own names stay in the engineers' tab and the PNG metadata)
PLAIN = {
    "Male": "Man",
    "Eyeglasses": "Glasses",
    "Hat": "Hat",
    "Goatee": "Goatee",
    "Sideburns": "Sideburns",
    "Beard": "Beard",
    "Lipstick": "Lipstick",
    "Gray_Hair": "Grey hair",
    "Bangs": "Bangs",
    "Blond_Hair": "Blond hair",
    "Smiling": "Smile",
    "Mouth_Open": "Open mouth",
}
FAST, BEST = "⚡ Instant", "✨ Best quality"
DECODE_CHUNK = 4  # faces per decoder call
# how a "Must NOT have" tick reads under a face
NOT = {"Man": "not a man", "Smile": "no smile", "Open mouth": "mouth closed"}
HOW = (  # the header's four-step strip: (image, title, one line)
    ("1_static.jpg", "1 · Random static", "Every face starts as pure noise."),
    ("2_cleaning.jpg", "2 · Cleaning up", "The AI removes a little noise, 25 times."),
    ("3_face.jpg", "3 · A new face", "Someone who has never existed."),
    ("4_choices.jpg", "4 · Your choices", "Add glasses, a smile, grey hair…"),
)
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)
CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#intro h1 { font-size: 2.3rem; margin-bottom: .3rem; }
#intro p { font-size: 1.12rem; line-height: 1.55; }
#how { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-top: 6px; }
#how figure { margin: 0; }
#how img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 14px; display: block; }
#how figcaption { margin-top: 8px; font-size: .95rem; line-height: 1.35; }
#how figcaption b { display: block; font-size: 1.02rem; }
button[role="tab"] { font-size: 1.05rem !important; padding: 10px 16px !important; }
.lead { font-size: 1.08rem; }
#make-btn, #make-btn button { font-size: 1.25rem !important; min-height: 56px; }
#verdict { font-size: 1.35rem; }
/* the checker's ticks under each face: wrap instead of cutting off */
.caption-label { white-space: normal !important; text-overflow: clip !important; max-width: 94% !important; line-height: 1.3; }
"""


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


def new_seed() -> int:
    return random.randrange(2**31)


def build_ui(e: Engine) -> gr.Blocks:
    hw = torch.cuda.get_device_name() if e.dev.type == "cuda" else "CPU"
    names = [n for n in e.attr_names]
    plain = [PLAIN.get(n, n.replace("_", " ")) for n in names]
    to_name = dict(zip(plain, names, strict=True))
    qualities = ([FAST] if e.student is not None else []) + ([BEST] if e.teacher is not None else [])
    to_mode = {FAST: e.fast_mode, BEST: e.teacher_mode}

    def choices_from(must: list[str], never: list[str]) -> list[str]:
        """'Must have' / 'Must not have' lists -> the engine's per-attribute any / yes / no (must have wins)."""
        return ["yes" if p in must else "no" if p in never else "any" for p in plain]

    def seen(p, choices: list[str]) -> list[str]:
        """Per face, the checker's (the DINOv3 probe's) verdict in plain words: a tick or a cross for each feature that
        was asked for or ruled out; if nothing was, the features it sees."""
        if p is None:
            return []
        out = []
        for row in p:
            yes = [
                ("✓ " if v > 0.5 else "✗ ") + pl.lower()
                for pl, c, v in zip(plain, choices, row, strict=True)
                if c == "yes"
            ]
            no = [
                ("✓ " if v <= 0.5 else "✗ ") + NOT.get(pl, "no " + pl.lower())
                for pl, c, v in zip(plain, choices, row, strict=True)
                if c == "no"
            ]
            marks = yes + no  # what was asked for first, then what was ruled out
            got = [pl.lower() for pl, v in zip(plain, row, strict=True) if v > 0.5]
            out.append(" · ".join(marks) if marks else "sees: " + (" · ".join(got[:3]) or "none of the 12"))
        return out

    def weak_note(quality: str, asked: list[str]) -> str:
        weak = [n for n in asked if n in e.student_weak] if to_mode[quality] == e.fast_mode else []
        if not weak:
            return ""
        what = ", ".join(PLAIN.get(n, n).lower() for n in weak)
        return f"⚠️ Instant mode is bad at {what} (measured: it draws them in under half the faces). Try **{BEST}**."

    reel = ASSETS / "showreel.mp4"
    with gr.Blocks(title="HumanDiT: faces of people who don't exist") as demo:
        # ------------------------------------------------------------ header
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, elem_id="intro"):
                gr.Markdown(
                    "# HumanDiT: faces of people who don't exist\n"
                    "This AI invents realistic faces from scratch. **None of these people are real.** Choose the "
                    "features you want and it draws new faces in about a second. Every picture carries an invisible "
                    "watermark, so it can always be identified as AI-made."
                )
                gr.HTML(
                    '<div id="how">'
                    + "".join(
                        f'<figure><img src="{data_uri(ASSETS / "how" / f)}" alt="{title}">'
                        f"<figcaption><b>{title}</b>{line}</figcaption></figure>"
                        for f, title, line in HOW
                        if (ASSETS / "how" / f).exists()
                    )
                    + "</div>"
                )
            if reel.exists():
                with gr.Column(scale=2, min_width=320):
                    gr.Video(
                        str(reel),
                        autoplay=True,
                        loop=True,
                        show_label=False,
                        interactive=False,
                        elem_id="reel",
                    )
                    gr.Markdown(
                        "<sub>The 15-second tour above is made entirely from this AI's own pictures.</sub>"
                    )

        with gr.Tabs():
            # -------------------------------------------------------- 1. make faces
            with gr.Tab("🎨 Make faces"):
                gr.Markdown(
                    "Tick any features you want (or leave everything blank and let the AI decide), then press "
                    "**Make faces**. Each press invents new people. Under each face, a second AI (the *checker*) "
                    "marks ✓ what came out as you asked, and ✗ what didn't.",
                    elem_classes="lead",
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        quality = gr.Radio(
                            qualities,
                            value=qualities[0],
                            label="Speed or quality?",
                            info="Instant: under a second. Best quality: a few seconds, more realistic.",
                        )
                        count = gr.Slider(1, 8, value=4, step=1, label="How many faces?")
                        must = gr.CheckboxGroup(plain, label="Must have")
                        never = gr.CheckboxGroup(plain, label="Must NOT have")
                        go = gr.Button("✨ Make faces", variant="primary", size="lg", elem_id="make-btn")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(
                            interactive=False,
                            label="Invented faces (not real people)",
                            columns=4,
                            height=560,
                            object_fit="cover",
                        )
                        timing = gr.Markdown()
                with gr.Accordion("⚙️ For engineers: sampler settings and the probe's scores", open=False):
                    with gr.Row():
                        seed = gr.Number(value=42, precision=0, label="Seed (the noise the faces start from)")
                        keep = gr.Checkbox(False, label="Keep the same people (reuse this seed)")
                    with gr.Row():
                        steps = gr.Slider(4, 100, value=TEACHER_STEPS, step=1, label="Teacher ODE steps")
                        cfg = gr.Slider(1.0, 6.0, value=2.0, step=0.1, label="Attribute CFG (teacher)")
                        ig = gr.Slider(
                            1.0, 3.0, value=IG_TUNED, step=0.01, label="Internal Guidance (teacher)"
                        )
                    probe_out = gr.Dataframe(
                        label="DINOv3 probe: P(attribute present) per face", interactive=False
                    )

                def generate(quality_, count_, must_, never_, seed_, keep_, steps_, cfg_, ig_):
                    seed_ = int(seed_) if keep_ else new_seed()
                    n, mode_, choices = int(count_), to_mode[quality_], choices_from(must_, never_)
                    noise, a = e.noise(seed_, n), e.attrs(choices, n)
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
                    asked = {nm: c for nm, c in zip(names, choices, strict=True) if c != "any"}
                    paths = [
                        e.save_png(
                            im,
                            f"{mode_}; seed {seed_} face {i}; steps {int(steps_)}; cfg {cfg_}; ig {ig_}; "
                            f"attributes {asked or 'unconditional'}",
                        )
                        for i, im in enumerate(imgs)
                    ]
                    msg = (
                        f"Made **{n} face{'s' * (n > 1)}** in **{t3 - t0:.1f} s** on this computer ({hw}).\n\n"
                        f"<sub>seed {seed_} · generator {1000 * (t1 - t0) / n:.0f} ms/face · decoder "
                        f"{1000 * (t2 - t1) / n:.0f} ms/face · watermark {1000 * (t3 - t2) / n:.0f} ms/face</sub>"
                    )
                    note = weak_note(quality_, [nm for nm, c in asked.items() if c == "yes"])
                    msg += f"\n\n{note}" if note else ""
                    table = (
                        None
                        if p is None
                        else {
                            "headers": ["face", *(pl for pl in plain)],
                            "data": [[i + 1, *(f"{v:.0%}" for v in row)] for i, row in enumerate(p)],
                        }
                    )
                    captions = seen(p, choices) or [None] * n
                    return list(zip(paths, captions, strict=True)), msg, table, seed_

                make_io = (
                    [quality, count, must, never, seed, keep, steps, cfg, ig],
                    [gallery, timing, probe_out, seed],
                )
                go.click(generate, *make_io, concurrency_id="gpu")
                demo.load(generate, *make_io, concurrency_id="gpu")  # the page opens with faces already made

            # -------------------------------------------------------- 2. change one thing
            with gr.Tab("✏️ Change one thing") as t_edit:
                gr.Markdown(
                    "The **same invented person**, three ways: **without** a feature, **the AI's own choice**, and "
                    "**with** it. Pick a feature, or try another person.",
                    elem_classes="lead",
                )
                with gr.Row():
                    e_attr = gr.Dropdown(
                        plain, value=PLAIN.get("Eyeglasses", plain[0]), label="Feature to change"
                    )
                    e_quality = gr.Radio(qualities, value=qualities[-1], label="Speed or quality?")
                    e_go = gr.Button("Show me", variant="primary")
                    e_new = gr.Button("🎲 Another person")
                with gr.Row():
                    e_no = gr.Image(label="Without", interactive=False)
                    e_any = gr.Image(label="The AI's choice", interactive=False)
                    e_yes = gr.Image(label="With", interactive=False)
                e_note = gr.Markdown()
                with gr.Accordion("⚙️ For engineers", open=False):
                    gr.Markdown(
                        "Same noise, one attribute token set to *no*, *null* (unspecified) and *yes*; everything "
                        "else unspecified. Identity is encouraged by the shared noise, not guaranteed."
                    )
                    with gr.Row():
                        e_seed = gr.Number(value=4, precision=0, label="Seed")
                        e_cfg = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Attribute CFG (teacher)")

                def edit(attr_, quality_, seed_, cfg_):
                    name, mode_ = to_name[attr_], to_mode[quality_]
                    j = names.index(name)
                    noise = e.noise(seed_, 1).repeat(3, 1, 1, 1)
                    a = torch.zeros(3, e.k, dtype=torch.long, device=e.dev)
                    a[:, j] = torch.tensor(
                        [AttributeTokens.NO, AttributeTokens.NULL, AttributeTokens.YES], device=e.dev
                    )
                    z, _ = e.sample(noise, a, mode_, TEACHER_STEPS, cfg_, IG_TUNED)
                    return *(
                        e.save_png(im, f"{mode_}; seed {int(seed_)}; {name} {state}; cfg {cfg_}")
                        for im, state in zip(e.decode(z), ("absent", "any", "present"), strict=True)
                    ), weak_note(quality_, [name])

                e_go.click(
                    edit,
                    [e_attr, e_quality, e_seed, e_cfg],
                    [e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )
                e_new.click(new_seed, None, e_seed).then(
                    edit,
                    [e_attr, e_quality, e_seed, e_cfg],
                    [e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )
                t_edit.select(
                    lambda cur, *args: (gr.skip(),) * 4 if cur is not None else edit(*args),
                    [e_no, e_attr, e_quality, e_seed, e_cfg],
                    [e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 3. blend two faces
            with gr.Tab("🎞️ Blend two faces") as t_morph:
                gr.Markdown(
                    "Watch one invented person **smoothly turn into another**, and back again.",
                    elem_classes="lead",
                )
                with gr.Row():
                    m_quality = gr.Radio(qualities, value=qualities[0], label="Speed or quality?")
                    m_go = gr.Button("🎲 Blend two new people", variant="primary")
                m_video = gr.Video(label="Blend", autoplay=True, loop=True, height=420, interactive=False)
                with gr.Accordion("⚙️ For engineers", open=False):
                    gr.Markdown(
                        "Spherical interpolation (slerp) between the two seeds' Gaussian noise tensors."
                    )
                    with gr.Row():
                        m_a = gr.Number(value=1, precision=0, label="Seed A")
                        m_b = gr.Number(value=2, precision=0, label="Seed B")
                        m_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")

                def morph(a_, b_, frames_, quality_):
                    import imageio.v3 as iio

                    mode_ = to_mode[quality_]
                    za, zb = e.noise(a_), e.noise(b_)
                    if int(a_) == int(b_):
                        zb = e.noise(int(b_) + 1)  # identical seeds would make the slerp angle 0
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

                m_go.click(lambda: (new_seed(), new_seed()), None, [m_a, m_b]).then(
                    morph, [m_a, m_b, m_frames, m_quality], m_video, concurrency_id="gpu"
                )
                t_morph.select(
                    lambda cur, *args: gr.skip() if cur is not None else morph(*args),
                    [m_video, m_a, m_b, m_frames, m_quality],
                    m_video,
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 4. watch it draw
            with gr.Tab("🔍 Watch it draw") as t_draw:
                gr.Markdown(
                    "The AI starts from **pure static** and removes a little of it in each of "
                    f"**{TEACHER_STEPS} steps** until a face appears. These are its drafts along the way.",
                    elem_classes="lead",
                )
                d_go = gr.Button("🎲 Draw a new person", variant="primary")
                with gr.Row():
                    d_video = gr.Video(
                        label="The whole drawing, step by step",
                        autoplay=True,
                        loop=True,
                        height=460,
                        interactive=False,
                    )
                    d_gallery = gr.Gallery(
                        label="Six of the drafts", columns=3, height=460, interactive=False
                    )
                with gr.Accordion("⚙️ For engineers", open=False):
                    gr.Markdown(
                        f"The teacher's flow-matching ODE ({TEACHER_STEPS} steps, Internal Guidance {IG_TUNED}). Each "
                        "frame is the clean-latent prediction x̂₀ at that step, decoded by the RAEv2 decoder."
                    )
                    d_seed = gr.Number(value=3, precision=0, label="Seed")

                def draw(seed_):
                    import imageio.v3 as iio

                    if e.teacher is None:
                        return None, []
                    _, traj = e.sample(
                        e.noise(seed_),
                        e.attrs([]),
                        e.teacher_mode,
                        TEACHER_STEPS,
                        1.0,
                        IG_TUNED,
                        trajectory=True,
                    )
                    imgs = []
                    for i in range(0, len(traj), 5):
                        imgs += e.decode(torch.cat(traj[i : i + 5]).float())
                    frames = [np.asarray(im) for im in imgs]
                    path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name  # noqa: SIM115
                    iio.imwrite(path, np.stack(frames + [frames[-1]] * 12), fps=8, codec="libx264")
                    pick = sorted({round(x) for x in np.linspace(0, len(imgs) - 1, 6)})
                    words = {0: "mostly static", len(imgs) - 1: "finished"}
                    return path, [
                        (imgs[i], f"Step {i + 1} of {len(imgs)}" + (f": {words[i]}" if i in words else ""))
                        for i in pick
                    ]

                d_go.click(new_seed, None, d_seed).then(
                    draw, d_seed, [d_video, d_gallery], concurrency_id="gpu"
                )
                t_draw.select(
                    lambda cur, s: (gr.skip(), gr.skip()) if cur else draw(s),
                    [d_gallery, d_seed],
                    [d_video, d_gallery],
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 5. instant or best?
            with gr.Tab("⚡ Instant or best?") as t_cmp:
                gr.Markdown(
                    "The same starting static, drawn four ways. **Instant** takes one step; **best quality** takes "
                    "more clean-up steps: slower, but closer to a real photo. The *realism gap* is measured on "
                    "thousands of faces: lower is better.",
                    elem_classes="lead",
                )
                with gr.Row():
                    c_attr = gr.Dropdown(
                        ["No particular feature", *plain],
                        value="No particular feature",
                        label="Ask for a feature",
                    )
                    c_go = gr.Button("Compare", variant="primary")
                    c_new = gr.Button("🎲 Another person")
                c_gallery = gr.Gallery(
                    columns=4, height=340, label="Same start, four ways", interactive=False
                )
                c_gaps = gr.Markdown()
                with gr.Accordion("⚙️ For engineers", open=False):
                    gr.Markdown(
                        "Times are live on this machine (sampling only). *Realism gap* is FID (clean) vs FFHQ-256, "
                        "at IG 1.55: FID-50k where it was measured (student, teacher at 25 and 50 steps), FID-10k "
                        "for the 10-step teacher (10k reads higher than 50k)."
                    )
                    c_seed = gr.Number(value=11, precision=0, label="Seed")

                def compare(seed_, attr_):
                    noise, a = e.noise(seed_), e.attrs([])
                    if attr_ in to_name:
                        a[:, names.index(to_name[attr_])] = AttributeTokens.YES
                    runs = [(e.fast_mode, 1, e.student_eval)] if e.student is not None else []
                    if e.teacher is not None:
                        runs += [
                            (e.teacher_mode, s, next((st for st in stems if st in e.evals), None))
                            for s, stems in TEACHER_EVALS
                        ]
                    out, gaps = [], []
                    for mode_, s, stem in runs:
                        e.sync()
                        t0 = time.perf_counter()
                        z, _ = e.sample(noise, a, mode_, s, 2.0, IG_TUNED)
                        e.sync()
                        sec = time.perf_counter() - t0
                        name = "Instant" if mode_ == e.fast_mode else f"Best quality · {s} steps"
                        out.append(
                            (
                                e.save_png(e.decode(z)[0], f"{mode_}; {s} steps; seed {int(seed_)}; {attr_}"),
                                f"{name} · {sec:.2f} s",
                            )
                        )
                        if stem in e.evals:
                            r, short = e.evals[stem], name.replace("Best quality · ", "")
                            gaps.append(
                                f"{short} **{r['fid_clean']:.1f}**"
                                + (
                                    ""
                                    if r["n"] == 50_000
                                    else f" (from {r['n']:,} faces, which reads a bit higher)"
                                )
                            )
                    note = (
                        "**Realism gap** (lower is better; each measured on 50,000 faces unless noted): "
                        + " · ".join(gaps)
                        if gaps
                        else ""
                    )
                    return out, note

                c_go.click(compare, [c_seed, c_attr], [c_gallery, c_gaps], concurrency_id="gpu")
                c_new.click(new_seed, None, c_seed).then(
                    compare, [c_seed, c_attr], [c_gallery, c_gaps], concurrency_id="gpu"
                )
                t_cmp.select(
                    lambda cur, *args: (gr.skip(), gr.skip()) if cur else compare(*args),
                    [c_gallery, c_seed, c_attr],
                    [c_gallery, c_gaps],
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 6. is it AI-made?
            with gr.Tab("🛡️ Is it AI-made?"):
                gr.Markdown(
                    "Every face this site makes carries a **hidden watermark** you can't see. Drop any picture here to "
                    "check for it: it survives resizing and JPEG compression. Or try one of the two examples below.",
                    elem_classes="lead",
                )
                with gr.Row():
                    w_in = gr.Image(type="pil", label="Your picture", height=360)
                    w_out = gr.Markdown(elem_id="verdict")
                examples = [
                    [str(p)]
                    for p in (
                        ASSETS / "examples" / "humandit_face.png",
                        ASSETS / "examples" / "old_project_face.png",
                    )
                    if p.exists()
                ]
                if examples:
                    gr.Examples(
                        examples,
                        inputs=w_in,
                        label="Examples: a HumanDiT face, and a face from this project's first version (no watermark)",
                    )
                w_in.change(
                    lambda im: (
                        ""
                        if im is None
                        else "✅ **Yes: this picture carries the HumanDiT watermark.** It was made by this AI."
                        if check_watermark(im, e.wm_device)
                        else "❌ **No HumanDiT watermark found.** This picture was not made here (or it was heavily "
                        "altered)."
                    ),
                    w_in,
                    w_out,
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 7. for engineers
            with gr.Tab("📊 For engineers"):
                gr.Markdown(
                    "**Teacher:** a DiT^DH-B flow-matching transformer (183M parameters) in the RAEv2 latent space "
                    "(frozen DINOv3-L features + a ViT-XL pixel decoder), trained with CMuon; sampled with a "
                    f"{TEACHER_STEPS}-step ODE and Internal Guidance {IG_TUNED} "
                    f"({e.score(f'teacher_tuned_{TEACHER_STEPS}step')}). **Instant mode:** the one-step "
                    f"{e.student_name} ({e.student_score}). **Checker:** a DINOv3 attribute probe. "
                    f"**Watermark:** TrustMark (device: {e.wm_device}). Running on **{hw}**"
                    + (" with ONNX Runtime." if e.ort_dec is not None else ".")
                    + " Methods, every measurement and what broke: README.md, MODEL_CARD.md and docs/."
                )
                weak = ", ".join(n.replace("_", " ") for n in e.student_weak)
                if weak:
                    gr.Markdown(
                        f"The {e.student_name} follows **{weak}** less than half the time (measured)."
                    )
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
                        "headers": [
                            "run",
                            "steps",
                            "samples",
                            "FID (clean)",
                            "FD_DINOv2",
                            "precision",
                            "recall",
                        ],
                        "data": rows,
                    }
                    if rows
                    else None,
                    label="Measured metrics (runs/eval*; scripts/evaluate.py)",
                    interactive=False,
                )
                pairs = [
                    p
                    for n in ("teacher_tuned", "teacher")
                    if (p := e.eval_dir / f"{n}_nearest_pairs.png").exists()
                ]
                if pairs:
                    gr.Image(
                        str(pairs[0]),
                        label="Memorisation audit: the most similar generated (top) vs training (bottom) faces",
                    )

        gr.Markdown(
            "Every face here is invented by an AI; none of them are real people. For research and education only, "
            "**not for commercial use**. It was trained on FFHQ (CC BY-NC-SA 4.0) with RAEv2 (CC BY-NC 4.0) and "
            "DINOv3 (DINOv3 License) components. See MODEL_CARD.md for limitations and bias."
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
    engine.warm_up()
    build_ui(engine).launch(
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
        head=REEL_MUTED,
        allowed_paths=[str(ASSETS)],
    )


if __name__ == "__main__":
    main()
