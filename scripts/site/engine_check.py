"""Call every endpoint of the website's engine (app/api.py) and check the answers' shape, sizes and timings.

    uv run python scripts/site/engine_check.py [--url http://localhost:7861]

Prints one line per endpoint and exits non-zero if any fails. Nothing is written anywhere.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[2]


def kb(uri: str) -> float:
    return len(uri) / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7861")
    a = ap.parse_args()
    c = Client(a.url, verbose=False)
    fails = []

    def run(name: str, check, **kw):
        t0 = time.perf_counter()
        try:
            r = c.predict(api_name=f"/{name}", **kw)
            ok, note = check(r)
        except Exception as err:  # the check fails, the run goes on
            ok, note, r = False, f"{type(err).__name__}: {err}", None
        print(
            f"{'PASS' if ok else 'FAIL'}  /{name:8s} {time.perf_counter() - t0:6.2f} s round trip  {note}",
            flush=True,
        )
        if not ok:
            fails.append(name)
        return r

    info = run(
        "ping", lambda r: (r["ok"] and len(r["features"]) == 12, f"{r['qualities']} on {r['hardware']}")
    )
    feats = info["features"] if info else ["Glasses", "Smile"]
    run(
        "make",
        lambda r: (
            len(r["faces"]) == 4
            and all(f["png"].startswith("data:image/png") for f in r["faces"])
            and all(len(f["marks"]) == 2 for f in r["faces"]),
            f"{r['seconds']} s engine, {sum(kb(f['png']) for f in r['faces']):.0f} KB, "
            f"marks {[m['ok'] for m in r['faces'][0]['marks']]}",
        ),
        count=4,
        quality="instant",
        must=[feats[1]],
        never=[feats[0]],
        seed=-1,
    )
    run(
        "make",
        lambda r: (len(r["faces"]) == 2, f"best: {r['seconds']} s engine, weak {r['weak']}"),
        count=2,
        quality="best",
        must=[],
        never=[],
        seed=7,
    )
    run(
        "edit",
        lambda r: (
            all(r[k].startswith("data:image/png") for k in ("without", "choice", "with")),
            f"{r['seconds']} s, checker {r['checker']}",
        ),
        feature_name="Glasses",
        quality="best",
        seed=4,
    )
    run(
        "blend",
        lambda r: (r["n"] == 24 and r["cols"] == 8, f"{r['seconds']} s, sprite {kb(r['src']):.0f} KB"),
        quality="instant",
        seed_a=1,
        seed_b=2,
        frames=24,
    )
    run(
        "draw",
        lambda r: (
            r["n"] == 25 and len(r["noise_left"]) == 25,
            f"{r['seconds']} s, sprite {kb(r['src']):.0f} KB, noise {r['noise_left'][0]}..{r['noise_left'][-1]}",
        ),
        seed=3,
    )
    run(
        "compare",
        lambda r: (
            len(r["runs"]) == 4,
            ", ".join(f"{x['label']} {x['seconds']:.2f}s fid {x['fid']}" for x in r["runs"]),
        ),
        seed=11,
        feature_name="",
    )
    for name, want in (("humandit_face.png", True), ("old_project_face.png", False)):
        uri = (
            "data:image/png;base64,"
            + base64.b64encode((ROOT / "docs/assets/examples" / name).read_bytes()).decode()
        )
        run("check", lambda r, w=want, n=name: (r["found"] is w, f"{n}: found={r['found']}"), image=uri)
    print(f"\n{'all endpoints pass' if not fails else 'FAILED: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
