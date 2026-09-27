"""Assemble the Hugging Face upload folders: the demo Space and the model repo it downloads its weights from.

    uv run python scripts/build_space.py --model-repo USER/HumanDiT --github https://github.com/USER/REPO

Writes dist/model (weights, probe, measured results, model card) and dist/space (the Gradio app, the humangen
package, the page's assets, requirements and a Space card). Nothing is uploaded: that is a separate, deliberate step,

    hf upload USER/HumanDiT dist/model . --repo-type model
    hf upload USER/HumanDiT-demo dist/space . --repo-type space

after which the Space's hardware is set to ZeroGPU in its settings. Large files are hard-linked when possible, so the
1.5 GB of weights are not copied twice on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

WEIGHTS = {  # model-repo path -> local path (directory names are kept: the demo keys its measured scores on them)
    "teacher_b/ema.safetensors": "runs/teacher_b/ckpt/ema.safetensors",
    "student_gan_fixed/student.safetensors": "runs/student_gan_fixed/student.safetensors",
    "attributes/attribute_probe.pt": "runs/attributes/attribute_probe.pt",
}
EVAL_DIRS = ("runs/eval", "runs/eval_extra", "runs/eval_local")
EVAL_EXTRA = ("teacher_tuned_nearest_pairs.png",)  # the memorisation audit image shown in the engineers' tab
SPACE_ASSETS = ("showreel_poster.jpg", "web", "examples")  # the page plays web/reel.mp4, not the 1080p master


def put(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def put_tree(src: Path, dst: Path) -> None:
    for f in sorted(src.rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            put(f, dst / f.relative_to(src))


def front_matter(fields: dict) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines += [f"{k}:"] + [f"  - {x}" for x in v]
        else:  # strings quoted (so "3.12" stays a string), booleans in YAML spelling
            lines.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join([*lines, "---", ""])


def absolute_links(md: str, github: str | None) -> str:
    """Relative links in the model card point into the GitHub repo; on the Hub they would 404."""
    if not github:
        return md
    base = github.rstrip("/")

    def fix(m: re.Match) -> str:
        text, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|#|mailto:)", target):
            return m.group(0)
        kind = "raw/main" if re.search(r"\.(png|jpe?g|webp|gif|mp4)$", target) else "blob/main"
        return f"[{text}]({base}/{kind}/{target})"

    md = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", fix, md)
    return re.sub(
        r'(src|href)="(?!https?:)([^"]+)"', lambda m: f'{m.group(1)}="{base}/raw/main/{m.group(2)}"', md
    )


def build_model(out: Path, github: str | None) -> None:
    for dst, src in WEIGHTS.items():
        put(ROOT / src, out / dst)
    seen = set()
    for d in EVAL_DIRS:
        for f in sorted((ROOT / d).glob("*.json")):
            assert f.name not in seen, f"{f.name} is in two eval directories"
            seen.add(f.name)
            put(f, out / "eval" / f.name)
    for name in EVAL_EXTRA:
        if (ROOT / "runs/eval" / name).exists():
            put(ROOT / "runs/eval" / name, out / "eval" / name)
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    meta = front_matter(
        {
            "license": "other",
            "license_name": "non-commercial",
            "license_link": "LICENSE.md",
            "library_name": "pytorch",
            "pipeline_tag": "unconditional-image-generation",
            "tags": ["diffusion", "flow-matching", "dit", "gan", "distillation", "faces", "synthetic-data"],
        }
    )
    (out / "README.md").write_text(meta + absolute_links(card, github), encoding="utf-8")
    (out / "LICENSE.md").write_text(
        "# Licence of these weights\n\n"
        "Non-commercial research and education only. The weights inherit the terms of what they were trained on "
        "and with: FFHQ (CC BY-NC-SA 4.0), RAEv2 (CC BY-NC 4.0) and DINOv3 (DINOv3 License). The training and demo "
        "code is MIT, except files adapted from RAEv2, which keep CC BY-NC 4.0.\n",
        encoding="utf-8",
    )


def build_space(out: Path, model_repo: str, github: str | None) -> None:
    put_tree(ROOT / "src" / "humangen", out / "src" / "humangen")
    for f in ("gradio_app.py", "watermark.py"):
        put(ROOT / "app" / f, out / "app" / f)
    put_tree(ROOT / "app" / "web", out / "app" / "web")
    for name in SPACE_ASSETS:
        src = ROOT / "docs" / "assets" / name
        put_tree(src, out / "docs" / "assets" / name) if src.is_dir() else put(
            src, out / "docs" / "assets" / name
        )
    (out / "app.py").write_text(
        (ROOT / "space" / "app.py").read_text(encoding="utf-8").replace("__MODEL_REPO__", model_repo),
        encoding="utf-8",
    )
    put(ROOT / "space" / "requirements.txt", out / "requirements.txt")
    put(ROOT / "LICENSE", out / "LICENSE")
    import gradio

    meta = front_matter(
        {
            "title": "HumanDiT",
            "colorFrom": "blue",
            "colorTo": "pink",
            "sdk": "gradio",
            "sdk_version": gradio.__version__,
            "python_version": "3.12",
            "app_file": "app.py",
            "pinned": False,
            "license": "other",
            "short_description": "An AI that invents faces of people who don't exist",
            "models": [model_repo],
        }
    )
    code = f" Code and full write-up: {github}." if github else ""
    (out / "README.md").write_text(
        meta + "# HumanDiT: faces of people who don't exist\n\n"
        "An AI that invents realistic faces from scratch, with features you choose. Every picture carries an "
        "invisible watermark. Runs on ZeroGPU: a GPU is attached only while your faces are being made."
        f"{code}\n\nWeights and measured results: https://huggingface.co/{model_repo}. Non-commercial use only.\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-repo", required=True, help="USER/NAME of the model repo the Space downloads from"
    )
    ap.add_argument("--github", help="https://github.com/USER/REPO, for links in the cards")
    ap.add_argument("--out", default="dist")
    a = ap.parse_args()
    out = ROOT / a.out
    for sub in ("model", "space"):
        shutil.rmtree(out / sub, ignore_errors=True)
    build_model(out / "model", a.github)
    build_space(out / "space", a.model_repo, a.github)
    for sub in ("model", "space"):
        files = [f for f in (out / sub).rglob("*") if f.is_file()]
        mb = sum(f.stat().st_size for f in files) / 2**20
        print(f"{out / sub}: {len(files)} files, {mb:.1f} MB")


if __name__ == "__main__":
    main()
