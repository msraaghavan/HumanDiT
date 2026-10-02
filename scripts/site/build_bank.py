"""A bank of saved results for the website's saved-results mode (site/assets/bank.json + site/assets/bank/).

    uv run --extra demo python app/api.py --port 7861 &      # the engine, on this machine's GPU
    uv run python scripts/site/build_bank.py                  # about half an hour on a GTX 1650 Ti

Until a visitor switches live mode on (the AI downloading to and running on their own device), the page answers
from this bank: "Make faces" picks faces whose checker verdicts match what was ticked, "Race
again" moves to another saved race. The bank
is uncurated: every output of every call is kept, made by the same endpoints the live page calls, with fixed seeds.
Nothing is selected by eye.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import time
from pathlib import Path

from gradio_client import Client
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
OUT = SITE / "assets" / "bank"
PLAN = {
    # faces: (quality, feature asked for or None, faces) -- unconditional faces cover "must not" requests
    "faces": [("instant", None, 240), ("best", None, 120), ("instant", "*", 16), ("best", "*", 8)],
    "compare_seeds": (51, 52, 53, 54, 55),
}
SEED0 = {"instant": 100_000, "best": 200_000}  # face seeds: a fixed range per quality, one call of 8 per seed


def save(uri: str, path: Path, quality: int = 84) -> str:
    im = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
    path = path.with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=quality, optimize=True, progressive=True)
    return path.relative_to(SITE).as_posix()


def raw(uri: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(uri.split(",", 1)[1]))
    return path.relative_to(SITE).as_posix()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7861")
    a = ap.parse_args()
    c = Client(a.url, verbose=False)
    info = c.predict(api_name="/ping")
    features = info["features"]
    shutil.rmtree(OUT, ignore_errors=True)
    t_start = time.time()

    faces = {"instant": [], "best": []}
    seconds = {"instant": [0.0, 0], "best": [0.0, 0]}
    jobs = []
    for quality, feat, n in PLAN["faces"]:
        for f in features if feat == "*" else [feat]:
            jobs += [(quality, f, min(8, n - k)) for k in range(0, n, 8)]
    for quality, feat, n in jobs:
        seed = SEED0[quality] + len(faces[quality])
        r = c.predict(
            count=n, quality=quality, must=[feat] if feat else [], never=[], seed=seed, api_name="/make"
        )
        seconds[quality][0] += r["seconds"]
        seconds[quality][1] += len(r["faces"])
        for f in r["faces"]:
            k = len(faces[quality])
            faces[quality].append(
                {
                    "src": save(f["png"], OUT / f"{quality[0]}{k:04d}"),
                    "sees": f["sees"],
                    "asked": feat or "",
                }
            )
        print(f"faces {quality:7s} {feat!s:11s} {len(faces[quality]):4d}  {r['seconds']:.2f} s", flush=True)

    compares = []
    for s in PLAN["compare_seeds"]:
        r = c.predict(seed=s, feature_name="", api_name="/compare")
        compares.append(
            {
                "seed": r["seed"],
                "hardware": r["hardware"],
                "runs": [
                    {**x, "img": save(x["img"], OUT / f"compare_{s}_{x['steps']}", 88)} for x in r["runs"]
                ],
            }
        )
        print(f"compare {s}: {[x['seconds'] for x in r['runs']]}", flush=True)

    bank = {
        "made_on": info["hardware"],
        "features": features,
        "faces": faces,
        "sec_per_face": {q: round(t / max(1, n), 3) for q, (t, n) in seconds.items()},
        "compares": compares,
    }
    (SITE / "assets" / "bank.json").write_text(json.dumps(bank, separators=(",", ":")))
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(
        f"bank: {len(faces['instant'])} + {len(faces['best'])} faces, "
        f"{len(compares)} races; {len(files)} files, "
        f"{sum(f.stat().st_size for f in files) / 2**20:.1f} MB; {(time.time() - t_start) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
