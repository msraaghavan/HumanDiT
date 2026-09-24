"""Download FFHQ-256 / CelebA from the Hugging Face Hub and cache RAE latents.

    # laptop smoke (first FFHQ shard, 1000 images)
    uv run python scripts/prepare_data.py ffhq --shards 1 --limit 1000
    # pod: full FFHQ with flips, and CelebA pooled features + attributes (all splits)
    python scripts/prepare_data.py ffhq
    python scripts/prepare_data.py celeba
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from humangen.data.cache import CacheSpec, build_cache
from humangen.rae import RAE, ensure_weights
from humangen.utils.device import get_precision

FFHQ_REPO = "merkol/ffhq-256"
CELEBA_REPO = "flwrlabs/celeba"
CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
    "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
    "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
    "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
    "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young",
]  # fmt: skip


def fetch(repo: str, prefix: str, local_dir: str, max_shards: int | None) -> list[Path]:
    files = sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet")
    )
    files = files[:max_shards] if max_shards else files
    return [Path(hf_hub_download(repo, f, repo_type="dataset", local_dir=local_dir)) for f in files]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["ffhq", "celeba"])
    ap.add_argument("--shards", type=int, default=None, help="max parquet shards to use (per split)")
    ap.add_argument("--limit", type=int, default=None, help="max images (per split)")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    prec = get_precision()
    ensure_weights()
    rae = RAE(load_decoder=False).to(prec.device)
    batch = args.batch or (64 if prec.autocast_dtype is not None else 8)
    # Encoding is always fp32 (TF32 on Ampere+): the latent statistics were fit on full-precision
    # features, and the cache is reused for every later stage.
    if args.dataset == "ffhq":
        shards = fetch(FFHQ_REPO, "data/", f"{args.data_dir}/ffhq-256", args.shards)
        spec = CacheSpec("ffhq256", shards, flips=True)
        build_cache(spec, rae, args.cache_dir, batch, args.workers, args.limit)
    else:
        for split in ("train", "valid", "test"):
            shards = fetch(CELEBA_REPO, f"img_align+identity+attr/{split}-", f"{args.data_dir}/celeba", args.shards)
            spec = CacheSpec(
                f"celeba_{split}", shards, flips=False, store_latents=False, label_columns=CELEBA_ATTRS, center_crop=True
            )
            build_cache(spec, rae, args.cache_dir, batch, args.workers, args.limit)


if __name__ == "__main__":
    main()
