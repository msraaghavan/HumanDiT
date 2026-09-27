"""Render the 15-second reel (scripts/showreel/reel.html) to video, frame-accurately, with motion blur and sound.

    uv run --extra demo python scripts/showreel/render.py                    # full render -> docs/assets/showreel.*
    uv run --extra demo python scripts/showreel/render.py --sheet 1,3.5,6.2  # contact sheet of single frames

The page draws every frame as a pure function of time, so each output frame is the average of --blur subframes spread
over a 180-degree shutter (real motion blur, not a filter). Frames are piped as raw RGB into ffmpeg (a near-lossless
master), then encoded for the web:
- docs/assets/showreel.mp4         1920x1080 H.264 with the synthesised soundtrack (scripts/showreel/audio.py)
- docs/assets/showreel.webp        960x540 animated WebP, silent, loops: for READMEs and model cards
- docs/assets/showreel_poster.jpg  a still of the end card
"""

from __future__ import annotations

import argparse
import functools
import http.server
import io
import subprocess
import sys
import threading
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio import synth  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS, SECONDS = 1920, 1080, 30, 15


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def grab(page, t: float) -> np.ndarray:
    page.evaluate(f"window.renderAt({t!r})")
    return np.asarray(Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB"))


def ffmpeg(args: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args], **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur", type=int, default=4, help="subframes per frame (1 = no motion blur)")
    ap.add_argument("--shutter", type=float, default=0.5, help="fraction of a frame the shutter stays open")
    ap.add_argument(
        "--sheet", default="", help="comma-separated times: write a contact sheet instead of a video"
    )
    ap.add_argument("--work", default="runs/showreel")
    ap.add_argument(
        "--channel", default="msedge", help="installed Chromium browser to drive (as demo_screenshots.py)"
    )
    ap.add_argument("--out", default="docs/assets")
    a = ap.parse_args()
    work, out = ROOT / a.work, ROOT / a.out
    work.mkdir(parents=True, exist_ok=True)
    srv, port = serve()
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=a.channel)
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.on("console", lambda m: print("[page]", m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: print("[page error]", e))
        page.goto(f"http://127.0.0.1:{port}/scripts/showreel/reel.html?render")
        page.wait_for_function("window.CUES !== undefined", timeout=120_000)  # set once assets are loaded

        if a.sheet:
            ts = [float(x) for x in a.sheet.split(",")]
            thumbs = [Image.fromarray(grab(page, t)).resize((640, 360), Image.LANCZOS) for t in ts]
            cols = min(3, len(thumbs))
            sheet = Image.new("RGB", (cols * 640, -(-len(thumbs) // cols) * 360))
            for i, im in enumerate(thumbs):
                sheet.paste(im, ((i % cols) * 640, (i // cols) * 360))
            sheet.save(work / "sheet.png")
            print("wrote", work / "sheet.png")
            browser.close()
            srv.shutdown()
            return

        cues = page.evaluate("window.CUES")
        impacts = [c["t"] for c in cues if c["kind"] == "impact"]
        master = work / "master.mp4"
        enc = ffmpeg(
            ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "10", "-pix_fmt", "yuv444p", str(master)],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        n, t0 = SECONDS * FPS, time.perf_counter()
        offsets = [(k + 0.5) / a.blur * a.shutter - a.shutter / 2 for k in range(a.blur)]
        for f in range(n):
            acc = np.zeros((H, W, 3), np.float32)
            for o in offsets:
                acc += grab(page, (f + o) / FPS)
            frame = (acc / len(offsets)).round().astype(np.uint8)
            hit = max((1 - (f / FPS - ti) / 0.13 for ti in impacts if 0 <= f / FPS - ti < 0.13), default=0.0)
            if hit > 0:  # RGB split on the cuts: red and blue pulled apart for four frames
                k = round(7 * hit)
                frame[..., 0] = np.roll(frame[..., 0], k, axis=1)
                frame[..., 2] = np.roll(frame[..., 2], -k, axis=1)
            enc.stdin.write(frame.tobytes())
            if f % 30 == 29:
                el = time.perf_counter() - t0
                print(
                    f"[render] {f + 1}/{n} frames, {el:.0f} s, eta {el / (f + 1) * (n - f - 1):.0f} s",
                    flush=True,
                )
        enc.stdin.close()
        enc.wait()
        browser.close()
    srv.shutdown()

    wav = work / "audio.wav"
    synth(cues, SECONDS, wav)
    out.mkdir(parents=True, exist_ok=True)
    mp4, webp, poster = out / "showreel.mp4", out / "showreel.webp", out / "showreel_poster.jpg"
    ffmpeg(
        ["-i", str(master), "-i", str(wav), "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-profile:v", "high",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(mp4)]
    ).wait()  # fmt: skip
    ffmpeg(
        ["-i", str(master), "-vf", "scale=960:540:flags=lanczos", "-c:v", "libwebp_anim", "-quality", "72",
         "-compression_level", "6", "-loop", "0", "-an", str(webp)]
    ).wait()  # fmt: skip
    ffmpeg(["-ss", "13.95", "-i", str(master), "-frames:v", "1", "-q:v", "2", str(poster)]).wait()
    for p in (mp4, webp, poster):
        print(f"[render] {p.relative_to(ROOT)}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
