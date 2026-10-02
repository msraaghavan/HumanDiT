"""The faces that bounce around the "this needs a laptop" screen: site/assets/gate.jpg.

    uv run python scripts/site/build_gate.py

Nine of the opening's faces (assets/hero.json, all made by the AI), each in four of its real drafts, from nearly done
back to the blur of the first steps: one row a face, one column a draft. Phones, tablets and computers that can't run
the AI see only this screen (js/gate.js), so it is small: about 120 KB.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
PICK = [9, 11, 17, 2, 12, 15, 7, 13, 19]  # hero.json order: the shocked, the cheeky and the startled first
DRAFTS = [24, 16, 9, 2]  # the finished face, then further and further back
SIZE = 144


def main() -> None:
    hero = json.loads((SITE / "assets/hero.json").read_text(encoding="utf-8"))
    out = Image.new("RGB", (SIZE * len(DRAFTS), SIZE * len(PICK)))
    for row, i in enumerate(PICK):
        f = hero["faces"][i]
        sheet = Image.open(SITE / f["src"]).convert("RGB")
        z, cols = f["size"], f["cols"]
        for col, d in enumerate(DRAFTS):
            d = min(d, f["n"] - 1)
            box = (d % cols * z, d // cols * z, d % cols * z + z, d // cols * z + z)
            out.paste(sheet.crop(box).resize((SIZE, SIZE), Image.LANCZOS), (col * SIZE, row * SIZE))
    dest = SITE / "assets/gate.jpg"
    out.save(dest, quality=80, optimize=True, progressive=True)
    meta = {"src": "assets/gate.jpg", "size": SIZE, "faces": len(PICK), "drafts": len(DRAFTS)}
    print(json.dumps(meta), f"{dest.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
