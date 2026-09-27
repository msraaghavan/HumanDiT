"""The engine behind the HumanDiT website: JSON endpoints that the static site (site/) calls, on a Hugging Face ZeroGPU
Space or on this machine.

    uv run --extra demo python app/api.py                 # http://localhost:7860, for site/ served on localhost

The website is static and hosted on its own (Cloudflare Pages); this process only makes pictures. Every endpoint
returns plain JSON. Faces people may download come back as PNG data URIs that carry the invisible watermark and
"AI-generated" metadata; sequences (drawing, blending) come back as one JPEG sprite sheet, frames left to right and
top to bottom, which the page scrubs through. Timings are measured around finished GPU work (torch.cuda.synchronize).
"""

import argparse
import base64
import io
import math
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# gradio_app imports `spaces` (ZeroGPU) before anything imports torch
import gradio as gr
import gradio_app as ga
import torch
from PIL import Image
from watermark import check_watermark, provenance_pnginfo

from humangen.models.layers import AttributeTokens

gpu, ZERO_GPU = ga.gpu, ga.ZERO_GPU
QUALITIES = {"instant": "Instant", "best": "Best quality"}
MAX_UPLOAD = 12 * 2**20  # bytes of an uploaded picture, after base64 decoding


def png_uri(img: Image.Image, details: str) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=provenance_pnginfo(details))
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jpeg_uri(img: Image.Image, quality: int = 88) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def sprite(imgs: list[Image.Image], cols: int, quality: int = 84) -> dict:
    size = imgs[0].width
    rows = math.ceil(len(imgs) / cols)
    sheet = Image.new("RGB", (cols * size, rows * size))
    for i, im in enumerate(imgs):
        sheet.paste(im.convert("RGB"), ((i % cols) * size, (i // cols) * size))
    return {"src": jpeg_uri(sheet, quality), "n": len(imgs), "cols": cols, "size": size}


def seed_or_new(seed: int) -> int:
    return int(seed) if int(seed) >= 0 else random.randrange(2**31)


def build(e: ga.Engine, site_url: str) -> gr.Blocks:
    names = list(e.attr_names)
    plain = [ga.PLAIN.get(n, n.replace("_", " ")) for n in names]
    index = {p.lower(): j for j, p in enumerate(plain)}
    modes = {k: {"Instant": e.fast_mode, "Best quality": e.teacher_mode}[v] for k, v in QUALITIES.items()}
    modes = {k: m for k, m in modes.items() if m in e.modes}
    hw_label = "Hugging Face ZeroGPU" if ZERO_GPU else e.gpu_name()
    limit = 4 if ZERO_GPU else 1  # ZeroGPU gives each job its own GPU slice; a laptop has one small GPU

    def mode_of(quality: str) -> str:
        if quality not in modes:
            raise gr.Error(f"Unknown quality {quality!r}; use one of {sorted(modes)}.")
        return modes[quality]

    def feature(name: str) -> int:
        j = index.get(name.strip().lower())
        if j is None:
            raise gr.Error(f"Unknown feature {name!r}.")
        return j

    def weak(quality: str, asked: list[int]) -> list[str]:
        if modes.get(quality) != e.fast_mode:
            return []
        return [plain[j] for j in asked if names[j] in e.student_weak]

    def where() -> str:
        return e.gpu_name() if e.dev.type == "cuda" else "the CPU"

    def timed(fn):
        e.sync()
        t0 = time.perf_counter()
        out = fn()
        e.sync()
        return out, time.perf_counter() - t0

    # ------------------------------------------------------------------ endpoints
    def ping() -> dict:
        """Is the engine up? Also what it can do: the 12 features, the two qualities, where it runs."""
        return {
            "ok": True,
            "features": plain,
            "qualities": sorted(modes),
            "weak": [ga.PLAIN.get(n, n) for n in e.student_weak],
            "hardware": hw_label,
            "steps": ga.TEACHER_STEPS,
        }

    def make_duration(count: int = 4, quality: str = "instant", *_a, **_k) -> int:
        return 20 if quality == "instant" else 30

    @gpu(duration=make_duration)
    def make(
        count: int = 4,
        quality: str = "instant",
        must: list[str] | None = None,
        never: list[str] | None = None,
        seed: int = -1,
    ) -> dict:
        """Invent `count` faces (1-8). `must` / `never`: feature names from /ping. seed -1: new people."""
        n, mode, seed = max(1, min(8, int(count))), mode_of(quality), seed_or_new(seed)
        choices = ["any"] * len(names)
        for f in never or []:
            choices[feature(f)] = "no"
        for f in must or []:  # "must have" wins over "must not"
            choices[feature(f)] = "yes"
        noise, a = e.noise(seed, n), e.attrs(choices, n)
        (z, _), t_gen = timed(lambda: e.sample(noise, a, mode, ga.TEACHER_STEPS, 2.0, ga.IG_TUNED))
        imgs, t_dec = timed(lambda: e.decode(z, mark=False))
        imgs, t_wm = timed(lambda: e.mark(imgs))
        p = e.probs(z)
        asked = {names[j]: c for j, c in enumerate(choices) if c != "any"}
        faces = []
        for i, im in enumerate(imgs):
            row = p[i] if p is not None else None
            marks = [
                {
                    "feature": plain[j],
                    "want": c == "yes",
                    "ok": bool((row[j] > 0.5) == (c == "yes")),
                    "p": round(float(row[j]), 3),
                }
                for j, c in enumerate(choices)
                if c != "any" and row is not None
            ]
            sees = [plain[j] for j in range(len(names)) if row is not None and row[j] > 0.5]
            details = (
                f"{mode}; seed {seed} face {i}; steps {ga.TEACHER_STEPS}; cfg 2.0; ig {ga.IG_TUNED}; "
                f"attributes {asked or 'unconditional'}"
            )
            faces.append({"png": png_uri(im, details), "marks": marks, "sees": sees})
        return {
            "seed": seed,
            "quality": quality,
            "faces": faces,
            "seconds": round(t_gen + t_dec + t_wm, 3),
            "ms_per_face": {
                k: round(1000 * v / n, 1) for k, v in (("make", t_gen), ("decode", t_dec), ("mark", t_wm))
            },
            "hardware": where(),
            "weak": weak(quality, [j for j, c in enumerate(choices) if c == "yes"]),
        }

    @gpu(duration=lambda feature_name="", quality="best", *_a, **_k: 20 if quality == "instant" else 30)
    def edit(feature_name: str = "Glasses", quality: str = "best", seed: int = -1) -> dict:
        """The same invented person without a feature, as the AI chooses, and with it."""
        j, mode, seed = feature(feature_name), mode_of(quality), seed_or_new(seed)
        noise = e.noise(seed, 1).repeat(3, 1, 1, 1)
        a = torch.zeros(3, e.k, dtype=torch.long, device=e.dev)
        a[:, j] = torch.tensor([AttributeTokens.NO, AttributeTokens.NULL, AttributeTokens.YES], device=e.dev)
        (z, _), sec = timed(lambda: e.sample(noise, a, mode, ga.TEACHER_STEPS, 3.0, ga.IG_TUNED))
        imgs = e.decode(z)
        p = e.probs(z)
        out = {
            k: png_uri(im, f"{mode}; seed {seed}; {names[j]} {state}; cfg 3.0")
            for k, im, state in zip(
                ("without", "choice", "with"), imgs, ("absent", "any", "present"), strict=True
            )
        }
        return {
            **out,
            "seed": seed,
            "feature": plain[j],
            "checker": None if p is None else [round(float(r[j]), 3) for r in p],
            "seconds": round(sec, 3),
            "hardware": where(),
            "weak": weak(quality, [j]),
        }

    @gpu(duration=lambda quality="instant", *_a, **_k: 20 if quality == "instant" else 30)
    def blend(quality: str = "instant", seed_a: int = -1, seed_b: int = -1, frames: int = 24) -> dict:
        """One invented person turning into another: slerp between two seeds' noise, as a sprite sheet."""
        mode, frames = mode_of(quality), max(8, min(32, int(frames)))
        sa, sb = seed_or_new(seed_a), seed_or_new(seed_b)
        if sa == sb:
            sb += 1  # identical seeds make the slerp angle 0
        za, zb = e.noise(sa), e.noise(sb)
        cos = ((za * zb).sum() / (za.norm() * zb.norm())).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.arccos(cos)
        ts = torch.linspace(0, 1, frames, device=e.dev).view(-1, 1, 1, 1)
        z0 = (torch.sin((1 - ts) * omega) * za + torch.sin(ts * omega) * zb) / torch.sin(omega)
        (lat, _), sec = timed(
            lambda: e.sample(z0, e.attrs([], frames), mode, ga.TEACHER_STEPS, 1.0, ga.IG_TUNED)
        )
        imgs = e.decode(lat)
        return {**sprite(imgs, 8), "seeds": [sa, sb], "seconds": round(sec, 3), "hardware": where()}

    @gpu(duration=25)
    def draw(seed: int = -1) -> dict:
        """The teacher's drafts: its clean-picture guess at each of its steps, and how much static is left then."""
        if e.teacher is None:
            raise gr.Error("This engine has no multi-step model.")
        seed = seed_or_new(seed)
        (out, sec) = timed(
            lambda: e.sample(
                e.noise(seed),
                e.attrs([]),
                e.teacher_mode,
                ga.TEACHER_STEPS,
                1.0,
                ga.IG_TUNED,
                trajectory=True,
            )
        )
        traj = out[1]
        imgs = []
        for i in range(0, len(traj), 5):
            imgs += e.decode(torch.cat(traj[i : i + 5]).float())
        t = e.transport.time_grid(ga.TEACHER_STEPS, torch.device("cpu"))[:-1].tolist()
        return {
            **sprite(imgs, 5),
            "seed": seed,
            "noise_left": [round(x, 4) for x in t],
            "seconds": round(sec, 3),
            "hardware": where(),
        }

    @gpu(duration=30)
    def compare(seed: int = -1, feature_name: str = "") -> dict:
        """The same starting noise through Instant (1 step) and the teacher at 10, 25 and 50 steps, timed live."""
        seed = seed_or_new(seed)
        noise, a = e.noise(seed), e.attrs([])
        if feature_name:
            a[:, feature(feature_name)] = AttributeTokens.YES
        runs = [(e.fast_mode, 1, e.student_eval)] if e.student is not None else []
        if e.teacher is not None:
            runs += [
                (e.teacher_mode, s, next((st for st in stems if st in e.evals), None))
                for s, stems in ga.TEACHER_EVALS
            ]
        out = []
        for mode, steps, stem in runs:
            (z, _), sec = timed(lambda m=mode, s=steps: e.sample(noise, a, m, s, 2.0, ga.IG_TUNED))
            r = e.evals.get(stem or "")
            out.append(
                {
                    "label": "Instant" if mode == e.fast_mode else f"{steps} steps",
                    "steps": steps,
                    "seconds": round(sec, 4),
                    "img": jpeg_uri(e.decode(z)[0], 90),
                    "fid": round(r["fid_clean"], 3) if r else None,
                    "fid_n": r["n"] if r else None,
                }
            )
        return {"seed": seed, "runs": out, "hardware": where()}

    @gpu(duration=15)
    def check(image: str) -> dict:
        """Does this picture (a data URI or base64) carry the HumanDiT watermark?"""
        raw = base64.b64decode(re.sub(r"^data:[^,]*,", "", image), validate=False)
        if len(raw) > MAX_UPLOAD:
            raise gr.Error("That picture is too large; try one under 12 MB.")
        try:
            im = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as err:
            raise gr.Error("That file is not a picture this can read.") from err
        return {"found": bool(check_watermark(im, e.wm_device)), "size": list(im.size)}

    link = (
        f'<p>The website is at <a href="{site_url}">{site_url}</a>.</p>'
        if site_url
        else "<p>The website is a separate static page that calls this engine.</p>"
    )
    with gr.Blocks(title="HumanDiT engine") as demo:
        gr.HTML(
            "<div style='max-width:640px;margin:48px auto;font:16px/1.6 system-ui;color:#e9f6ff'>"
            "<h1 style='font-size:28px;margin:0 0 8px'>HumanDiT engine</h1>"
            "<p>This Space runs the models behind HumanDiT, an AI that invents faces of people who don't exist. "
            f"It has no page of its own.</p>{link}</div>"
        )
        gr.api(ping, api_name="ping", concurrency_limit=None)
        for fn in (make, edit, blend, draw, compare, check):
            gr.api(fn, api_name=fn.__name__, concurrency_id="gpu", concurrency_limit=limit)
    return demo


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="runs/teacher_b/ckpt/ema.safetensors")
    ap.add_argument("--student", default="runs/student_gan_fixed/student.safetensors")
    ap.add_argument("--probe", default="runs/attributes/attribute_probe.pt")
    ap.add_argument("--eval-dirs", nargs="+", default=["runs/eval", "runs/eval_extra", "runs/eval_local"])
    ap.add_argument("--onnx-dir", default="onnx")
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    ap.add_argument("--site-url", default=os.environ.get("HUMANDIT_SITE_URL", ""))
    ap.add_argument("--no-warm-up", action="store_true")
    a = ap.parse_args(argv)
    engine = ga.Engine(a.teacher, a.student, a.probe, a.eval_dirs, a.onnx_dir)
    if not (a.no_warm_up or ZERO_GPU):
        engine.warm_up()
    build(engine, a.site_url).launch(
        server_port=a.port,
        theme=gr.themes.Base(),
        footer_links=["api"],
        ssr_mode=False,
        run_history=False,  # results are multi-megabyte data URIs; never keep them in a visitor's browser
    )


if __name__ == "__main__":
    main()
