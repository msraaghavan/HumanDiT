"""Pack the showreel's model outputs into the small files the demo page animates (docs/assets/web).

    uv run python scripts/showreel/build_assets.py     # first: renders the model outputs into runs/showreel/assets
    uv run python scripts/showreel/render.py           # then: renders docs/assets/showreel.mp4
    uv run python scripts/showreel/build_web.py

Writes sprite sheets (one JPEG per sequence, frames left to right, top to bottom), a 960x540 copy of the reel for
the page, and motion.json, which holds the sheet layouts, the latent-token colours and the measured numbers the page
counts up to. Nothing here is hand-made: every picture is a model output, every number comes from runs/eval via the
showreel manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image


def sheet(paths: list[Path], cols: int, size: int, out: Path, quality: int) -> dict:
    rows = math.ceil(len(paths) / cols)
    im = Image.new("RGB", (cols * size, rows * size))
    for i, p in enumerate(paths):
        tile = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        im.paste(tile, ((i % cols) * size, (i // cols) * size))
    im.save(out, quality=quality, optimize=True, progressive=True)
    return {"file": out.name, "n": len(paths), "cols": cols, "size": size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/showreel/assets")
    ap.add_argument("--out", default="docs/assets/web")
    ap.add_argument("--reel", default="docs/assets/showreel.mp4", help="the rendered reel (render.py)")
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((src / "manifest.json").read_text())

    traj = sorted(src.glob("traj_*.png"))
    edits = [src / f"edit_{i}.png" for i in range(len(m["edits"]))]
    faces = sorted(src.glob("grid_*.png"))
    motion = {
        # the teacher's drafts, step 1..25 (hero seed)
        "traj": sheet(traj, 5, 256, out / "traj.jpg", 84),
        # the hero face: no extras, then one more feature each
        "edits": sheet(edits, len(edits), 256, out / "edits.jpg", 86),
        # uncurated one-step faces, seeds 0..71
        "faces": sheet(faces, 12, 128, out / "faces.jpg", 80),
        "edit_probe": m["edits"],
        # 16x16 latent tokens as RGB (PCA of the final latent): noise, then each ODE step
        "tokens": m["tokens"],
        "metrics": m["metrics"],
        "speed": m["speed"],
        "gpu": m["gpu"],
        "hero_seed": m["hero_seed"],
        "grid_seeds": m["grid_seeds"],
    }
    (out / "motion.json").write_text(json.dumps(motion, separators=(",", ":")))
    reel = Path(a.reel)
    if reel.exists():  # the page's copy of the reel: decoding the 1080p master in a small monitor cost frames
        import subprocess

        import imageio_ffmpeg

        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(reel),
                "-vf",
                "scale=960:540",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                str(out / "reel.mp4"),
            ],
            check=True,
        )
    for f in sorted(out.iterdir()):
        print(f"{f.name:12s} {f.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
