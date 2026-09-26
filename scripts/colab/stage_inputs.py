"""Collect everything a Colab VM needs into runs/colab/inputs (then: scripts/colab/upload_inputs.sh SESSION).

humandit-code.tar.gz          `git archive HEAD` (commit your changes first; a dirty tree is reported)
teacher_ema.safetensors       runs/teacher_b/ckpt/ema.safetensors (+ .sha256, checked after reassembly on the VM)
attribute_probe.pt, attributes.json   runs/attributes/
ffhq256_dinov2_real.npz       runs/eval/ (real FFHQ DINOv2 features for FD_DINOv2 / precision-recall)
fixed_images.npy              the first 32 images of the first local FFHQ shard (scripts/platform_check.py)
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

from humangen.data.parquet_images import decode_image, iter_rows, list_shards

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "colab" / "inputs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True
    )
    subprocess.check_call(
        ["git", "archive", "--format=tar.gz", "-o", str(OUT / "humandit-code.tar.gz"), "HEAD"], cwd=REPO
    )
    print(f"code: {head}{' (WARNING: uncommitted changes are NOT in the archive)' if dirty.strip() else ''}")
    for src, name in (
        ("runs/teacher_b/ckpt/ema.safetensors", "teacher_ema.safetensors"),
        ("runs/attributes/attribute_probe.pt", "attribute_probe.pt"),
        ("runs/attributes/attributes.json", "attributes.json"),
        ("runs/eval/ffhq256_dinov2_real.npz", "ffhq256_dinov2_real.npz"),
    ):
        shutil.copy2(REPO / src, OUT / name)
    h = hashlib.sha256()
    with open(OUT / "teacher_ema.safetensors", "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    (OUT / "teacher_ema.safetensors.sha256").write_text(h.hexdigest())
    rows = next(iter_rows(list_shards(REPO / "data" / "ffhq-256" / "data")[:1], ["image"], batch_rows=32))
    np.save(OUT / "fixed_images.npy", np.stack([decode_image(c) for c in rows["image"][:32]]))
    for p in sorted(OUT.iterdir()):
        print(f"{p.stat().st_size / 1e6:9.1f} MB  {p.name}")


if __name__ == "__main__":
    main()
