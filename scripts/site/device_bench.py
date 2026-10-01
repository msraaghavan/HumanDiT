"""Measure how fast the website's AI runs on this machine's graphics chips and processor, through the real page.

    uv run python scripts/site/serve.py --port 8081      # in another terminal (or the humandit-site-csp preview)
    uv run python scripts/site/device_bench.py --gpu "NVIDIA GeForce GTX 1650 Ti (laptop)" \
        --igpu "Intel UHD Graphics (built in)" --cpu "Intel Core i7-10750H, 12 threads"

Each device runs in headless Edge: models loaded, one face to warm up (shaders compile on first use), then 4 Instant
faces and 1 Best-quality face from fixed seeds, timed by the page itself around finished work. Writes
site/assets/speeds.json, which the page shows next to the visitor's own measured speed. Only devices measured here
appear on the site; re-measuring one device keeps the others' rows.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[2]

JS = """
async (best) => {
  const e = await import('/js/engine.js');
  await e.goLive();
  await e.call('make', { count: 1, quality: 'instant', seed: 1 });
  const r = await e.call('make', { count: 4, quality: 'instant', seed: 2 });
  const out = { instant_s: r.seconds / 4, split_ms: r.ms_per_face, hardware: r.hardware };
  if (best) {
    await e.call('make', { count: 1, quality: 'best', seed: 3 });
    const b = await e.call('make', { count: 1, quality: 'best', seed: 4 });
    out.best_s = b.seconds;
  }
  return out;
}
"""


async def run(url: str, kind: str, best: bool) -> dict:
    args = ["--enable-unsafe-webgpu"] + (["--force_high_performance_gpu"] if kind == "gpu" else [])
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="msedge", headless=True, args=args)
        pg = await b.new_page()
        await pg.goto(url + ("?ai=cpu" if kind == "cpu" else ""))
        r = await pg.evaluate(JS, best)
        await b.close()
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    ap.add_argument("--gpu", help="name of the separate graphics card (WebGPU, high-performance adapter)")
    ap.add_argument("--igpu", help="name of the built-in graphics (WebGPU, default adapter)")
    ap.add_argument("--cpu", help="name of the processor (WebAssembly, no WebGPU)")
    ap.add_argument(
        "--no-best", action="store_true", help="skip Best quality on the processor (minutes a face)"
    )
    a = ap.parse_args()
    devices = []
    for kind, name in (("gpu", a.gpu), ("igpu", a.igpu), ("cpu", a.cpu)):
        if not name:
            continue
        r = asyncio.run(run(a.url, kind, not (kind == "cpu" and a.no_best)))
        r = {
            "kind": kind,
            "name": name,
            **{k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()},
        }
        print(json.dumps(r), flush=True)
        devices.append(r)
    out = ROOT / "site/assets/speeds.json"
    if out.exists():  # re-measuring one device keeps the others' rows
        kinds = {d["kind"] for d in devices}
        devices = [d for d in json.loads(out.read_text())["devices"] if d["kind"] not in kinds] + devices
        devices.sort(key=lambda d: ("gpu", "igpu", "cpu").index(d["kind"]))
    out.write_text(json.dumps({"measured": time.strftime("%Y-%m-%d"), "devices": devices}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
