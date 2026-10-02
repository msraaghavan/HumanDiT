"""The opening of the website: one large face at a time, each forming from static through the AI's own drafts.

    uv run --extra demo python app/api.py --port 7861 &      # the engine, on this machine's GPU
    uv run python scripts/site/build_hero.py                  # -> site/assets/hero/ + site/assets/hero.json

Each face is the teacher (Best quality, 25 steps) from a fixed seed, in seed order (none chosen for looking good), with its
clean-picture guess at every step saved as a 5 x 5 sheet of 256 px drafts. The page plays a sheet from static to the
finished face, then holds it; in live mode the visitor's own device makes the next face instead. Seeds in SKIP are left
out, each with its reason.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
from pathlib import Path

from gradio_client import Client
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "assets" / "hero"
SEEDS = range(300, 324)  # 24 faces, in order
SKIP = {305: "face paint that reads as blood: a poor first picture for the opening"}  # left out, and why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7861")
    a = ap.parse_args()
    c = Client(a.url, verbose=False)
    info = c.predict(api_name="/ping")
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    faces = []
    for s in (s for s in SEEDS if s not in SKIP):
        r = c.predict(seed=s, api_name="/draw")
        sheet = Image.open(io.BytesIO(base64.b64decode(r["src"].split(",", 1)[1]))).convert("RGB")
        path = OUT / f"draw_{s}.jpg"
        sheet.save(path, quality=86, optimize=True, progressive=True)
        faces.append(
            {
                "src": path.relative_to(ROOT / "site").as_posix(),
                **{k: r[k] for k in ("n", "cols", "size", "seed", "noise_left", "seconds")},
            }
        )
        print(f"seed {s}: {r['seconds']} s, {path.stat().st_size // 1024} KB", flush=True)
    (ROOT / "site" / "assets" / "hero.json").write_text(
        json.dumps({"made_on": info["hardware"], "faces": faces, "skipped": SKIP}, separators=(",", ":"))
    )
    total = sum(f.stat().st_size for f in OUT.iterdir())
    print(f"{len(faces)} faces, {total / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
