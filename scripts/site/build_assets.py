"""Pre-make everything the website (site/) shows before its engine answers, by calling the engine itself.

    uv run --extra demo python app/api.py --port 7861 &      # the engine, on this machine's GPU
    uv run python scripts/site/build_assets.py                # -> site/assets/

The page never starts empty: until a visitor switches live mode on (the AI running on their own device), each section
shows these results, made by the same models with fixed seeds and labelled with where they ran. Every
picture is a model output and every number is measured: timings from this run, scores from runs/eval*.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import shutil
from pathlib import Path

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "assets"
HERO_SEEDS = (4, 0, 1, 2)  # the showreel's hero seed first, then the next seeds in order (not picked by eye)
STUDIO_SEED = 42
COMPARE_SEED = 11
# the project's story in measured FID-50k (clean): (eval file, plain name, technical name, what the site uses it for)
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


def save_uri(uri: str, path: Path) -> str:
    """Write a data URI to a file; return its URL relative to site/."""
    head, data = uri.split(",", 1)
    ext = ".png" if "png" in head else ".jpg"
    path = path.with_suffix(ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    return path.relative_to(ROOT / "site").as_posix()


def to_jpeg(uri: str, path: Path, quality: int = 88) -> str:
    """Pre-made faces are shown, not downloaded: JPEG is a fifth of the PNG's size."""
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
    path = path.with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=quality, optimize=True, progressive=True)
    return path.relative_to(ROOT / "site").as_posix()


def evals() -> dict:
    out = {}
    for d in ("runs/eval", "runs/eval_extra", "runs/eval_local"):
        for f in sorted((ROOT / d).glob("*.json")):
            try:
                r = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
            if "fid_clean" in r:
                out[f.stem] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7861")
    a = ap.parse_args()
    c = Client(a.url, verbose=False)
    pre = OUT / "preset"
    shutil.rmtree(pre, ignore_errors=True)
    info = c.predict(api_name="/ping")
    print("engine:", info["hardware"])

    hero = []
    for s in HERO_SEEDS:
        r = c.predict(seed=s, api_name="/draw")
        hero.append(
            {
                "src": save_uri(r["src"], pre / f"draw_{s}"),
                **{k: r[k] for k in ("n", "cols", "size", "seed", "noise_left", "seconds")},
            }
        )
        print(f"draw seed {s}: {r['seconds']} s")

    r = c.predict(count=8, quality="instant", must=[], never=[], seed=STUDIO_SEED, api_name="/make")
    studio = {
        **{k: r[k] for k in ("seed", "quality", "seconds", "ms_per_face", "hardware", "weak")},
        "faces": [
            {"src": to_jpeg(f["png"], pre / f"make_{i}"), "marks": f["marks"], "sees": f["sees"]}
            for i, f in enumerate(r["faces"])
        ],
    }
    print(f"make x8 instant: {r['seconds']} s")

    r = c.predict(seed=COMPARE_SEED, feature_name="", api_name="/compare")
    compare = {
        "seed": r["seed"],
        "hardware": r["hardware"],
        "runs": [{**x, "img": to_jpeg(x["img"], pre / f"compare_{x['steps']}")} for x in r["runs"]],
    }
    print("compare:", [(x["label"], x["seconds"]) for x in r["runs"]])

    # the face wall and the watermark examples (made by scripts/showreel/*)
    web = ROOT / "docs" / "assets"
    motion = json.loads((web / "web" / "motion.json").read_text())
    for src, dst in (
        (web / "web" / "faces.jpg", OUT / "faces.jpg"),
        (web / "web" / "edits.jpg", OUT / "edits.jpg"),
        (web / "examples" / "humandit_face.png", OUT / "examples" / "humandit_face.png"),
        (web / "examples" / "old_project_face.png", OUT / "examples" / "old_project_face.png"),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    check = {}  # the engine's verdict on each watermark example, shown if the engine can't be reached
    for name in ("humandit_face.png", "old_project_face.png"):
        uri = "data:image/png;base64," + base64.b64encode((OUT / "examples" / name).read_bytes()).decode()
        check[f"assets/examples/{name}"] = c.predict(image=uri, api_name="/check")["found"]
    print("check:", check)

    ev = evals()
    rows = [
        {"name": name, "tech": tech, "use": use, "fid": round(ev[k]["fid_clean"], 3), "n": ev[k]["n"]}
        for k, name, tech, use in SCOREBOARD
        if k in ev
    ]
    top = math.log10(max(r["fid"] for r in rows) * 1.15)
    for r in rows:
        r["w"] = round(math.log10(max(r["fid"], 1.0)) / top, 4)
    site = {
        "features": info["features"],
        "weak": info["weak"],
        "steps": info["steps"],
        "metrics": motion["metrics"],
        "speed": motion["speed"],
        "speed_gpu": motion["gpu"],
        "faces": motion["faces"],
        "edit_story": {"sheet": "assets/edits.jpg", **motion["edits"], "probe": motion["edit_probe"]},
        "scoreboard": rows,
        "table": sorted(
            (
                {
                    "run": k,
                    "steps": r.get("steps"),
                    "n": r["n"],
                    "fid": r["fid_clean"],
                    "fd_dinov2": r.get("fd_dinov2"),
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                }
                for k, r in ev.items()
            ),
            key=lambda r: r["fid"],
        ),
        "preset": {
            "made_on": info["hardware"],
            "hero": hero,
            "studio": studio,
            "compare": compare,
            "check": check,
        },
    }
    (OUT / "site.json").write_text(json.dumps(site, separators=(",", ":")))
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"site/assets: {sum(1 for f in OUT.rglob('*') if f.is_file())} files, {total / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
