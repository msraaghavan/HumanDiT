"""HumanDiT live demo, written for anyone: make faces, change one feature, blend two faces, watch it draw, compare
instant vs best quality, check the watermark. Everything technical lives in one "For engineers" tab.

    uv run --extra demo python app/gradio_app.py
    uv run --extra demo python app/gradio_app.py --teacher runs/teacher_b/ckpt/ema.safetensors \
        --student runs/student_gan_fixed/student.safetensors --probe runs/attributes/attribute_probe.pt

On a Hugging Face ZeroGPU Space (space/app.py) a GPU is attached only while a job runs: every job below is wrapped in
`gpu(...)`, which is `spaces.GPU` there and does nothing anywhere else.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import quote

try:  # Hugging Face ZeroGPU; imported before torch touches CUDA
    import spaces

    ZERO_GPU = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "true"}  # set by Hugging Face
    gpu = spaces.GPU
except ImportError:
    ZERO_GPU = False

    def gpu(fn=None, **_):
        return fn if fn is not None else (lambda f: f)


import gradio as gr
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from watermark import add_watermark, check_watermark, load_watermarker, provenance_pnginfo

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
        self.chunk = chunk_size(self.dev)
        self.transport = Transport(shift=dimension_shift(1024 * 16 * 16))
        self.wm_device = (
            ("cuda" if self.dev.type == "cuda" else "cpu") if watermark_device == "auto" else watermark_device
        )
        try:  # load TrustMark now: on ZeroGPU, models must be placed at start-up, not inside a job
            load_watermarker(self.wm_device)
        except Exception as err:
            print(f"[demo] TrustMark on {self.wm_device} failed ({err}); watermarking on the CPU")
            self.wm_device = "cpu"
            load_watermarker("cpu")
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
        if not trajectory and len(noise) > self.chunk:
            parts = [
                self.sample(n, a, mode, steps, cfg, ig)[0]
                for n, a in zip(noise.split(self.chunk), attrs.split(self.chunk), strict=True)
            ]
            return torch.cat(parts), []
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
        for zc in z.split(self.chunk):
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
        noise, a = self.noise(0, self.chunk), self.attrs([], self.chunk)
        for mode in self.modes:
            z, _ = self.sample(noise, a, mode, 2, 2.0, IG_TUNED)
            self.decode(z)
            self.probs(z)
        self.sync()
        print(f"[demo] warmed up in {time.perf_counter() - t0:.1f} s")

    def gpu_name(self) -> str:
        """Call inside a job: on ZeroGPU the GPU exists only there."""
        if self.dev.type != "cuda":
            return "the CPU"
        try:
            return torch.cuda.get_device_name()
        except RuntimeError:
            return "a shared cloud GPU"

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
FAST, BEST = "Instant", "Best quality"


def chunk_size(dev: torch.device) -> int:
    """Faces per GPU call. Measured on the 4 GB GTX 1650 Ti (teacher, 25 steps, CFG on): 3.48 / 3.40 / 3.55 s a face
    at batches of 1 / 2 / 4, but 33 s at 8, where it overflows into shared system memory (the decoder: 4.0 s a face
    instead of 0.17). With a browser also holding video memory, a 3-face edit overflowed too, so small GPUs take 2
    faces per call, which costs nothing; large ones take 8."""
    if dev.type != "cuda" or ZERO_GPU:  # ZeroGPU: no GPU is attached at start-up, and it has 48 GB
        return 8
    try:
        return 2 if torch.cuda.get_device_properties(dev).total_memory < 8 * 2**30 else 8
    except RuntimeError:
        return 8


# how a "Must NOT have" tick reads under a face
NOT = {"Man": "not a man", "Smile": "no smile", "Open mouth": "mouth closed"}
# the how-it-works stage: (title, one line, the label on its animated picture before the script takes over)
HOW = (
    ("Random static", "Every face starts as pure noise.", "pure noise"),
    ("Cleaning up", f"The AI removes a little noise, {TEACHER_STEPS} times.", "step 01 / 25"),
    ("A new face", "Someone who has never existed.", "not a real person"),
    ("Your choices", "Add glasses, a smile, grey hair…", "no extras"),
)


def neon_theme() -> gr.themes.Base:
    """Dark neon: electric cyan and hot magenta on near-black. Light-mode variables get the same values as dark ones,
    so the page is dark whatever the visitor's system setting (head_html also asks Gradio for dark mode)."""
    cyan, text, dim, line, line2 = (
        "#00e5ff",
        "#e9f6ff",
        "#8d98b8",
        "rgba(0, 229, 255, 0.28)",
        "rgba(124, 92, 255, 0.22)",
    )
    panel, field = "rgba(11, 15, 30, 0.82)", "#080b16"
    glow = "0 0 22px rgba(0, 229, 255, 0.35)"
    v = {
        "body_background_fill": "#05060b",
        "body_text_color": text,
        "body_text_color_subdued": dim,
        "background_fill_primary": "#0a0d1a",
        "background_fill_secondary": "#070912",
        "border_color_primary": line2,
        "border_color_accent": cyan,
        "border_color_accent_subdued": line,
        "color_accent_soft": "rgba(0, 229, 255, 0.12)",
        "block_background_fill": panel,
        "block_border_color": line2,
        "block_border_width": "1px",
        "block_shadow": "none",
        "block_info_text_color": dim,
        "block_label_background_fill": "rgba(5, 6, 11, 0.9)",
        "block_label_border_color": line,
        "block_label_border_width": "1px",
        "block_label_text_color": cyan,
        "block_title_background_fill": "rgba(0, 229, 255, 0.1)",
        "block_title_border_color": line,
        "block_title_border_width": "0px",
        "block_title_text_color": cyan,
        "panel_background_fill": "#0a0d1a",
        "panel_border_color": line2,
        "panel_border_width": "1px",
        "accordion_text_color": text,
        "input_background_fill": field,
        "input_background_fill_focus": field,
        "input_background_fill_hover": "#0b0f1f",
        "input_border_color": line2,
        "input_border_color_focus": cyan,
        "input_border_color_hover": line,
        "input_border_width": "1px",
        "input_placeholder_color": dim,
        "input_shadow": "none",
        "input_shadow_focus": "0 0 0 1px #00e5ff, 0 0 16px rgba(0, 229, 255, 0.35)",
        "checkbox_background_color": field,
        "checkbox_background_color_hover": "#0b0f1f",
        "checkbox_background_color_focus": field,
        "checkbox_background_color_selected": cyan,
        "checkbox_border_color": "rgba(148, 163, 255, 0.4)",
        "checkbox_border_color_hover": cyan,
        "checkbox_border_color_focus": cyan,
        "checkbox_border_color_selected": cyan,
        "checkbox_border_width": "1px",
        "checkbox_label_background_fill": "rgba(13, 18, 34, 0.9)",
        "checkbox_label_background_fill_hover": "rgba(0, 229, 255, 0.08)",
        "checkbox_label_background_fill_selected": "rgba(0, 229, 255, 0.16)",
        "checkbox_label_border_color": line2,
        "checkbox_label_border_color_hover": line,
        "checkbox_label_border_color_selected": cyan,
        "checkbox_label_border_width": "1px",
        "checkbox_label_shadow": "none",
        "checkbox_label_shadow_hover": "none",
        "checkbox_label_shadow_active": "none",
        "checkbox_label_text_color": text,
        "checkbox_label_text_color_selected": "#ffffff",
        "slider_color": cyan,
        "loader_color": cyan,
        "button_border_width": "1px",
        "button_primary_background_fill": "linear-gradient(90deg, #00e5ff, #7c5cff 55%, #ff3df0)",
        "button_primary_background_fill_hover": "linear-gradient(90deg, #5cf0ff, #9a80ff 55%, #ff6ff4)",
        "button_primary_border_color": "rgba(0, 229, 255, 0.6)",
        "button_primary_border_color_hover": cyan,
        "button_primary_text_color": "#04050a",
        "button_primary_text_color_hover": "#04050a",
        "button_primary_shadow": glow,
        "button_primary_shadow_hover": "0 0 30px rgba(0, 229, 255, 0.55), 0 0 60px rgba(255, 61, 240, 0.25)",
        "button_primary_shadow_active": "0 0 12px rgba(0, 229, 255, 0.5)",
        "button_secondary_background_fill": "rgba(13, 18, 34, 0.9)",
        "button_secondary_background_fill_hover": "rgba(0, 229, 255, 0.1)",
        "button_secondary_border_color": line,
        "button_secondary_border_color_hover": cyan,
        "button_secondary_text_color": text,
        "button_secondary_text_color_hover": "#ffffff",
        "button_secondary_shadow": "none",
        "button_secondary_shadow_hover": "0 0 18px rgba(0, 229, 255, 0.3)",
        "button_secondary_shadow_active": "none",
        "table_border_color": line2,
        "table_even_background_fill": "#0a0e1a",
        "table_odd_background_fill": "#0d1222",
        "table_row_focus": "rgba(0, 229, 255, 0.08)",
        "table_text_color": text,
        "code_background_fill": "#0a0e1a",
        "link_text_color": cyan,
        "link_text_color_hover": "#5cf0ff",
        "link_text_color_active": cyan,
        "link_text_color_visited": "#b08cff",
        "stat_background_fill": "rgba(0, 229, 255, 0.2)",
        "shadow_spread": "0px",
        "error_background_fill": "rgba(255, 77, 109, 0.08)",
        "error_border_color": "#ff4d6d",
        "error_border_width": "1px",
        "error_icon_color": "#ff4d6d",
        "error_text_color": "#ff8fa3",
    }
    base = gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.fuchsia,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    )
    return base.set(**v, **{k + "_dark": x for k, x in v.items()})


THEME = neon_theme()
WEB_SRC = Path(__file__).resolve().parent / "web"  # the motion layer: motion.css + motion.js
WEB = ASSETS / "web"  # its model outputs and numbers (scripts/showreel/build_web.py)
CSS = (WEB_SRC / "motion.css").read_text(encoding="utf-8")
FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900'
    '&family=JetBrains+Mono:wght@500;600&display=swap">'
)


def file_url(path: Path) -> str:
    """URL of a file under `allowed_paths`, relative to the page, so it also works behind a proxy (HF Spaces)."""
    try:
        rel = Path(os.path.relpath(path, Path.cwd()))
    except ValueError:  # another drive on Windows
        rel = path
    return "gradio_api/file=" + quote(rel.as_posix())


def jpeg_uri(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def head_html() -> str:
    js = (WEB_SRC / "motion.js").read_text(encoding="utf-8")
    base = file_url(WEB) + "/"
    # dark whatever the visitor's system says: ask Gradio (?__theme=dark; on HF Spaces the Hub passes its own theme),
    # and hold its "dark" class on <body> (Gradio leaves it off when the system is in light mode)
    dark = (
        "<script>(() => { const u = new URL(location.href); if (u.searchParams.get('__theme') !== 'dark') "
        "{ u.searchParams.set('__theme', 'dark'); history.replaceState(history.state, '', u); } "
        "const hold = () => { if (!document.body.classList.contains('dark')) document.body.classList.add('dark'); }; "
        "const start = () => { hold(); new MutationObserver(hold).observe(document.body, "
        "{ attributes: true, attributeFilter: ['class'] }); }; "
        # Gradio inserts this <head> content after the page has loaded, so the load event may be long gone
        "if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start(); "
        "})();</script>"
    )
    return dark + FONTS + f"<script>window.HD_BASE = {json.dumps(base)};</script><script>{js}</script>"


def hero_html() -> str:
    """The page's opening: kinetic headline, measured numbers that count to their values, a drifting wall of
    invented faces, and the 15-second showreel in a 'monitor'. Readable as plain HTML if the script never runs."""
    m = json.loads((WEB / "motion.json").read_text()) if (WEB / "motion.json").exists() else None
    head = ["Faces", "of", "people"]
    hi = ["who", "don't", "exist."]
    h1 = " ".join(f'<span class="w" style="--i:{i}"><span>{w}</span></span>' for i, w in enumerate(head))
    h1 += (
        '<br><span class="hl">'
        + " ".join(
            f'<span class="w" style="--i:{len(head) + j};--j:{j}"><span>{html.escape(w)}</span></span>'
            for j, w in enumerate(hi)
        )
        + "</span>"
    )
    stats = ""
    if m:
        sp, mt = m["speed"], m["metrics"]
        copies = round(mt["memorised"] * mt["memo_n"])
        rows = (
            (
                f"{sp['student_ms'] / 1000:.2f}",
                "s",
                f'data-from="{sp["teacher_s"]:.2f}" data-dec="2"',
                f"per face in Instant mode, on a laptop GPU (careful mode: {sp['teacher_s']:.1f} s)",
            ),
            (
                f"{mt['fid_after']:.2f}",
                "",
                f'data-from="{mt["fid_before"]:.2f}" data-dec="2"',
                f"realism gap, lower is better (the first version: {mt['fid_before']:.1f})",
            ),
            (
                str(mt["n_attributes"]),
                "",
                'data-from="0" data-dec="0"',
                "features you can ask for, or rule out",
            ),
            (
                str(copies),
                f" / {mt['memo_n']:,}",
                "data-scramble",
                "invented faces that copy a real person's photo",
            ),
        )
        stats = (
            '<div class="hd-stats">'
            + "".join(
                f'<div class="hd-stat" style="--i:{i}"><b data-to="{v}" {attrs}>{v}<small>{unit}</small></b>'
                f"<span>{html.escape(label)}</span></div>"
                for i, (v, unit, attrs, label) in enumerate(rows)
            )
            + "</div>"
        )
    # the hero plays the 960x540 web encode (build_web.py): decoding the 1080p master cost the page frames
    reel = WEB / "reel.mp4" if (WEB / "reel.mp4").exists() else ASSETS / "showreel.mp4"
    poster = ASSETS / "showreel_poster.jpg"
    monitor = ""
    if reel.exists():
        monitor = (
            '<div class="hd-monitor"><div class="hd-box"><div class="hd-screen">'
            f'<video src="{file_url(reel)}" poster="{file_url(poster) if poster.exists() else ""}" autoplay muted loop '
            'playsinline preload="auto" aria-label="A 15-second showreel made from this AI\'s own pictures"></video>'
            '</div><span class="hd-corner tl"></span><span class="hd-corner tr"></span><span class="hd-corner bl">'
            '</span><span class="hd-corner br"></span></div>'
            '<div class="hd-hud"><span>Showreel</span><span class="hd-tc">00:00.0 / 00:15.0</span>'
            '<span class="hd-prog"><i></i></span><button class="hd-sound" type="button">Sound off</button></div>'
            '<div class="hd-cap">A 15-second tour, made entirely from this AI\'s own pictures.</div></div>'
        )
    return (
        '<section class="hd-hero"><canvas class="hd-wall" aria-hidden="true"></canvas><div class="hd-hero-grid">'
        '<div class="hd-copy"><div class="hd-over"><span class="hd-dot"></span>'
        '<span class="hd-over-t">HumanDiT · an AI that invents faces</span></div>'
        f'<h1 class="hd-h1">{h1}</h1>'
        '<p class="hd-sub">This AI invents realistic faces from scratch. <b>None of these people are real.</b> '
        "Choose the features you want and it draws new faces in about a second. Every picture carries an invisible "
        "watermark, so it can always be identified as AI-made.</p>"
        f"{stats}</div>{monitor}</div></section>"
    )


def how_html() -> str:
    dots = "".join(f'<b style="left:{100 * i / (len(HOW) - 1):.3f}%"></b>' for i in range(len(HOW)))
    cards = "".join(
        f'<figure class="hd-step"><div class="hd-frame"><canvas aria-hidden="true"></canvas>'
        f'<span class="hd-tag">{tag}</span></div>'
        f"<figcaption><b><em>{i + 1}</em>{title}</b>{line}</figcaption></figure>"
        for i, (title, line, tag) in enumerate(HOW)
    )
    return (
        '<section class="hd-how"><div class="hd-how-head"><span class="hd-kicker">How it works</span>'
        f"<h2>From static to a face in {TEACHER_STEPS} small steps</h2></div>"
        f'<div class="hd-steps"><div class="hd-rail"><i></i>{dots}</div>{cards}</div>'
        '<div class="hd-cue"><i></i>Try it yourself below<i></i></div></section>'
    )


def race_html(times: list[tuple[str, float]], gaps: list[tuple[str, float, int]], hw: str) -> str:
    """Instant or best?: how long each run took (bars that replay the live timings at their real speed) and each
    setting's measured realism gap."""
    tmax = max(sec for _, sec in times)
    lanes = "".join(
        f'<div class="hd-lane time" data-sec="{sec:.4f}"><span>{html.escape(label)}</span>'
        f'<div class="hd-track"><i style="transform:scaleX({sec / tmax:.4f})"></i></div><b>{sec:.2f} s</b></div>'
        for label, sec in times
    )
    out = f'<div class="hd-race"><section><h4>How long each took</h4><p>Live, on {html.escape(hw)}. The bars replay each one at its real speed.</p>{lanes}</section>'
    if gaps:
        gmax, best = max(f for _, f, _ in gaps), min(f for _, f, _ in gaps)
        small = any(n != 50_000 for _, _, n in gaps)
        glanes = "".join(
            f'<div class="hd-lane gapl" data-val="{f:.3f}" data-max="{gmax:.3f}"><span>{html.escape(label)}</span>'
            f'<div class="hd-track gap{" best" if f == best else ""}"><i style="transform:scaleX({f / gmax:.4f})">'
            f'</i></div><b data-suffix="{"*" if n != 50_000 else ""}">{f:.1f}{"*" if n != 50_000 else ""}</b></div>'
            for label, f, n in gaps
        )
        note = " *Measured on 10,000 faces, which reads a bit higher." if small else ""
        out += (
            "<section><h4>Realism gap</h4><p>How far the pictures are from real photos, lower is better. Measured on "
            f"50,000 faces each.{note}</p>{glanes}</section>"
        )
    return out + "</div>"


# the project's story in measured FID-50k (clean): (eval file, plain name, technical name, used by the demo as)
SCOREBOARD = (
    ("legacy_dcgan_64px", "The first version", "DCGAN, 64 px (the original repo)", ""),
    ("gan_1step", "One-step student, first try", "adversarial, collapsed (a timestep bug)", ""),
    ("gan_anchor_1step", "One-step student + anchors", "adversarial + teacher ODE pairs", ""),
    ("wflow_1step", "One-step student, W-Flow", "Wasserstein flow distillation", ""),
    ("gan_fixed_1step", "One-step student, fixed", "adversarial, timestep bug fixed", "Instant"),
    ("teacher", "Teacher, 50 steps", "DiT^DH-B, default guidance 1.78", ""),
    ("teacher_tuned", "Teacher, 50 steps, tuned", "Internal Guidance 1.55", ""),
    ("teacher_tuned_25step", "Teacher, 25 steps, tuned", "Internal Guidance 1.55", "Best quality"),
)


def scoreboard_html(evals: dict) -> str:
    """Log-scale bars (lower is better): FID spans 3 to 207, so a linear scale would hide the teacher rows."""
    import math

    rows = [(evals[k]["fid_clean"], name, tech, use) for k, name, tech, use in SCOREBOARD if k in evals]
    if not rows:
        return ""
    top = math.log10(max(f for f, *_ in rows) * 1.15)
    best = min(f for f, *_ in rows)
    out = "".join(
        f'<div class="hd-srow{" best" if f == best else " live" if use else ""}" data-v="{f:.3f}" '
        f'data-w="{math.log10(max(f, 1.0)) / top:.4f}"><span>{html.escape(name)}'
        f"{f' · <em>{use}</em> mode' if use else ''}<small>{html.escape(tech)}</small></span>"
        f'<div class="hd-track"><i style="transform:scaleX({math.log10(max(f, 1.0)) / top:.4f})"></i></div>'
        f"<b>{f:.2f}</b></div>"
        for f, name, tech, use in rows
    )
    return (
        '<div class="hd-score"><h4>Realism gap of every model, measured</h4><p>FID (clean) against FFHQ-256 on '
        "50,000 samples each, log scale, lower is better. Cyan to pink: what this page runs; green: the best.</p>"
        f"{out}</div>"
    )


def wipe_html(without: Image.Image, with_: Image.Image, feature: str) -> str:
    f = html.escape(feature.lower())
    return (
        f'<div class="hd-wipe"><img src="{jpeg_uri(without)}" alt="without {f}">'
        f'<img class="hd-after" src="{jpeg_uri(with_)}" alt="with {f}"><span class="hd-bar"></span>'
        f'<span class="hd-lbl l">Without</span><span class="hd-lbl r">With {f}</span></div>'
        f'<div class="hd-wipe-cap">Same starting static; the AI was asked for one change: {f}.</div>'
    )


def new_seed() -> int:
    return random.randrange(2**31)


def build_ui(e: Engine) -> gr.Blocks:
    # named once for the engineers' tab; jobs name the GPU they actually ran on
    hw = "Hugging Face ZeroGPU (a shared NVIDIA GPU, attached per job)" if ZERO_GPU else e.gpu_name()
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
        return f"**Note:** Instant mode is bad at {what} (measured: it draws them in under half the faces). Try **{BEST}**."

    with gr.Blocks(title="HumanDiT: faces of people who don't exist") as demo:
        # ------------------------------------------------------------ header: hero + how it works (app/web)
        gr.HTML(hero_html(), elem_id="hd-hero-block", js_on_load=None)
        gr.HTML(how_html(), elem_id="hd-how-block", js_on_load=None)

        with gr.Tabs(elem_id="hd-tabs"):
            # -------------------------------------------------------- 1. make faces
            with gr.Tab("Make faces"):
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
                        go = gr.Button("Make faces", variant="primary", size="lg", elem_id="make-btn")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(
                            interactive=False,
                            elem_classes=["hd-dev", "hd-load"],
                            label="Invented faces (not real people)",
                            columns=4,
                            height=560,
                            object_fit="cover",
                        )
                        timing = gr.Markdown(elem_id="timing")
                with gr.Accordion("For engineers: sampler settings and the probe's scores", open=False):
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

                @gpu(duration=30)
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
                    where = f"GPU: {e.gpu_name()}" if e.dev.type == "cuda" else "on the CPU"
                    msg = (
                        f"Made **{n} face{'s' * (n > 1)}** in **{t3 - t0:.1f} s** ({where}).\n\n"
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
            with gr.Tab("Change one thing") as t_edit:
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
                    e_new = gr.Button("Another person")
                e_wipe = gr.HTML(js_on_load=None, elem_classes="hd-load")
                with gr.Row():
                    e_no = gr.Image(
                        label="Without", interactive=False, height=260, elem_classes=["hd-dev", "hd-load"]
                    )
                    e_any = gr.Image(
                        label="The AI's choice",
                        interactive=False,
                        height=260,
                        elem_classes=["hd-dev", "hd-load"],
                    )
                    e_yes = gr.Image(
                        label="With", interactive=False, height=260, elem_classes=["hd-dev", "hd-load"]
                    )
                e_note = gr.Markdown()
                with gr.Accordion("For engineers", open=False):
                    gr.Markdown(
                        "Same noise, one attribute token set to *no*, *null* (unspecified) and *yes*; everything "
                        "else unspecified. Identity is encouraged by the shared noise, not guaranteed."
                    )
                    with gr.Row():
                        e_seed = gr.Number(value=4, precision=0, label="Seed")
                        e_cfg = gr.Slider(1.0, 6.0, value=3.0, step=0.1, label="Attribute CFG (teacher)")

                @gpu(duration=30)
                def edit(attr_, quality_, seed_, cfg_):
                    name, mode_ = to_name[attr_], to_mode[quality_]
                    j = names.index(name)
                    noise = e.noise(seed_, 1).repeat(3, 1, 1, 1)
                    a = torch.zeros(3, e.k, dtype=torch.long, device=e.dev)
                    a[:, j] = torch.tensor(
                        [AttributeTokens.NO, AttributeTokens.NULL, AttributeTokens.YES], device=e.dev
                    )
                    z, _ = e.sample(noise, a, mode_, TEACHER_STEPS, cfg_, IG_TUNED)
                    imgs = e.decode(z)
                    return (
                        wipe_html(imgs[0], imgs[2], attr_),
                        *(
                            e.save_png(im, f"{mode_}; seed {int(seed_)}; {name} {state}; cfg {cfg_}")
                            for im, state in zip(imgs, ("absent", "any", "present"), strict=True)
                        ),
                        weak_note(quality_, [name]),
                    )

                e_go.click(
                    edit,
                    [e_attr, e_quality, e_seed, e_cfg],
                    [e_wipe, e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )
                e_new.click(new_seed, None, e_seed).then(
                    edit,
                    [e_attr, e_quality, e_seed, e_cfg],
                    [e_wipe, e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )
                t_edit.select(
                    lambda cur, *args: (gr.skip(),) * 5 if cur is not None else edit(*args),
                    [e_no, e_attr, e_quality, e_seed, e_cfg],
                    [e_wipe, e_no, e_any, e_yes, e_note],
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 3. blend two faces
            with gr.Tab("Blend two faces") as t_morph:
                gr.Markdown(
                    "Watch one invented person **smoothly turn into another**, and back again.",
                    elem_classes="lead",
                )
                with gr.Row():
                    m_quality = gr.Radio(qualities, value=qualities[0], label="Speed or quality?")
                    m_go = gr.Button("Blend two new people", variant="primary")
                m_video = gr.Video(
                    label="Blend",
                    autoplay=True,
                    loop=True,
                    height=420,
                    interactive=False,
                    elem_id="blend-video",
                    elem_classes="hd-load",
                )
                with gr.Accordion("For engineers", open=False):
                    gr.Markdown(
                        "Spherical interpolation (slerp) between the two seeds' Gaussian noise tensors."
                    )
                    with gr.Row():
                        m_a = gr.Number(value=1, precision=0, label="Seed A")
                        m_b = gr.Number(value=2, precision=0, label="Seed B")
                        m_frames = gr.Slider(8, 48, value=24, step=1, label="Frames")

                @gpu(duration=60)
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
            with gr.Tab("Watch it draw") as t_draw:
                gr.Markdown(
                    "The AI starts from **pure static** and removes a little of it in each of "
                    f"**{TEACHER_STEPS} steps** until a face appears. These are its drafts along the way.",
                    elem_classes="lead",
                )
                d_go = gr.Button("Draw a new person", variant="primary")
                with gr.Row():
                    d_video = gr.Video(
                        label="The whole drawing, step by step",
                        autoplay=True,
                        loop=True,
                        height=460,
                        interactive=False,
                        elem_id="draw-video",
                        elem_classes="hd-load",
                    )
                    d_gallery = gr.Gallery(
                        label="Six of the drafts",
                        columns=3,
                        height=460,
                        interactive=False,
                        elem_classes=["hd-dev", "hd-load"],
                    )
                with gr.Accordion("For engineers", open=False):
                    gr.Markdown(
                        f"The teacher's flow-matching ODE ({TEACHER_STEPS} steps, Internal Guidance {IG_TUNED}). Each "
                        "frame is the clean-latent prediction x̂₀ at that step, decoded by the RAEv2 decoder."
                    )
                    d_seed = gr.Number(value=3, precision=0, label="Seed")

                @gpu(duration=30)
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
            with gr.Tab("Instant or best?") as t_cmp:
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
                    c_new = gr.Button("Another person")
                c_gallery = gr.Gallery(
                    columns=4,
                    height=340,
                    label="Same start, four ways",
                    interactive=False,
                    elem_classes=["hd-dev", "hd-load"],
                )
                c_gaps = gr.HTML(js_on_load=None)
                with gr.Accordion("For engineers", open=False):
                    gr.Markdown(
                        "Times are live on this machine (sampling only). *Realism gap* is FID (clean) vs FFHQ-256, "
                        "at IG 1.55: FID-50k where it was measured (student, teacher at 25 and 50 steps), FID-10k "
                        "for the 10-step teacher (10k reads higher than 50k)."
                    )
                    c_seed = gr.Number(value=11, precision=0, label="Seed")

                @gpu(duration=30)
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
                    out, times, gaps = [], [], []
                    for mode_, s, stem in runs:
                        e.sync()
                        t0 = time.perf_counter()
                        z, _ = e.sample(noise, a, mode_, s, 2.0, IG_TUNED)
                        e.sync()
                        sec = time.perf_counter() - t0
                        name = "Instant" if mode_ == e.fast_mode else f"Best quality · {s} steps"
                        short = name.replace("Best quality · ", "")
                        times.append((short, sec))
                        out.append(
                            (
                                e.save_png(e.decode(z)[0], f"{mode_}; {s} steps; seed {int(seed_)}; {attr_}"),
                                f"{name} · {sec:.2f} s",
                            )
                        )
                        if stem in e.evals:
                            gaps.append((short, e.evals[stem]["fid_clean"], e.evals[stem]["n"]))
                    return out, race_html(times, gaps, e.gpu_name())

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
            with gr.Tab("Is it AI-made?"):
                gr.Markdown(
                    "Every face this site makes carries a **hidden watermark** you can't see. Drop any picture here to "
                    "check for it: it survives resizing and JPEG compression. Or try one of the two examples below.",
                    elem_classes="lead",
                )
                with gr.Row():
                    w_in = gr.Image(type="pil", label="Your picture", height=360, elem_id="wm-in")
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

                @gpu(duration=15)
                def verdict(im):
                    if im is None:
                        return ""
                    if check_watermark(im, e.wm_device):
                        return "**Yes: this picture carries the HumanDiT watermark.** It was made by this AI."
                    return (
                        "**No HumanDiT watermark found.** This picture was not made here (or it was heavily "
                        "altered)."
                    )

                w_in.change(
                    verdict,
                    w_in,
                    w_out,
                    concurrency_id="gpu",
                )

            # -------------------------------------------------------- 7. for engineers
            with gr.Tab("For engineers"):
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
                gr.HTML(scoreboard_html(e.evals), js_on_load=None)
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--student", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--eval-dirs", nargs="+", default=["runs/eval", "runs/eval_extra", "runs/eval_local"])
    ap.add_argument("--onnx-dir", default="onnx", help="ONNX graphs used when no CUDA GPU is available")
    ap.add_argument("--watermark-device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    ap.add_argument("--share", action="store_true")
    ap.add_argument(
        "--no-warm-up", action="store_true", help="skip the start-up warm-up (ZeroGPU has no GPU then)"
    )
    args = ap.parse_args(argv)
    engine = Engine(
        args.teacher, args.student, args.probe, args.eval_dirs, args.onnx_dir, args.watermark_device
    )
    if not (args.no_warm_up or ZERO_GPU):
        engine.warm_up()
    build_ui(engine).launch(
        server_port=args.port,
        share=args.share,
        theme=THEME,
        css=CSS,
        head=head_html(),
        allowed_paths=[str(ASSETS)],
        footer_links=[],  # no "Use via API / Built with Gradio / Settings" footer
    )


if __name__ == "__main__":
    main()
