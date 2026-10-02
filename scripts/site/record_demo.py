"""The site's demo video (site/assets/demo.mp4 + demo_poster.jpg): the real page, used in a real browser, recorded.

    uv run python scripts/site/serve.py --port 8081 &      # the page and its models, under the production CSP
    uv run python scripts/site/record_demo.py [--url http://localhost:8081/]

Headless Edge on this machine's graphics card, in a fresh profile (so the one-time download is shown), drives the
page through its own buttons: switching live mode on, making faces with chosen features, changing one thing about a
person, the Instant-or-Best race and the watermark check. Every result and timing in the video is what the page
produced during the recording; nothing is staged or edited in. A visible cursor and captions are drawn into the page.
Frames come from the browser's compositor (Chrome DevTools screencast) and are laid onto a steady 30 fps timeline,
then encoded as H.264 with no sound. The two one-time waits (preparing the AI for the graphics card, and the Best
quality model's first download) play at 4x speed, with a "4x speed" badge on screen while they do.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.async_api import Page, async_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "assets"
W, H, SCALE = 1440, 810, 4 / 3  # CSS pixels; recorded at 1920 x 1080
FPS = 30
FAST = 4  # playback speed of the one-time waits, shown on screen

# a cursor and captions in the top layer (so they stay above the page's dialogs); styles via the DOM, which the CSP allows
OVERLAY = """() => {
  const top = (el) => { el.popover = 'manual'; document.body.append(el); el.showPopover(); return el; };
  const cur = document.createElement('div');
  cur.innerHTML = '<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2l15 11-6.6.9L9.6 21z" fill="#fff" stroke="#111" stroke-width="1.4" stroke-linejoin="round"/></svg>';
  Object.assign(cur.style, { position: 'fixed', inset: 'auto', left: '0', top: '0', margin: '0', padding: '0', border: '0',
    background: 'transparent', overflow: 'visible', pointerEvents: 'none', transform: 'translate(-100px, -100px)', zIndex: '9' });
  const ring = document.createElement('div');
  Object.assign(ring.style, { position: 'absolute', left: '-14px', top: '-14px', width: '28px', height: '28px',
    borderRadius: '50%', border: '2px solid #f2a65a', opacity: '0' });
  cur.prepend(ring);
  const cap = document.createElement('div');
  Object.assign(cap.style, { position: 'fixed', inset: 'auto', left: '50%', bottom: '44px', margin: '0', maxWidth: '980px',
    width: 'max-content', padding: '14px 22px', border: '1px solid rgba(255,255,255,.14)', borderRadius: '12px',
    background: 'rgba(11,12,15,.92)', color: '#fff', font: '500 21px/1.4 Inter, system-ui, sans-serif',
    letterSpacing: '-0.01em', textAlign: 'center', transform: 'translateX(-50%)', opacity: '0',
    transition: 'opacity .35s ease', pointerEvents: 'none' });
  const fast = document.createElement('div');
  fast.textContent = '4× speed';
  Object.assign(fast.style, { position: 'fixed', inset: 'auto', right: '24px', top: '22px', margin: '0', padding: '7px 12px',
    border: '1px solid rgba(242,166,90,.5)', borderRadius: '999px', background: 'rgba(11,12,15,.92)', color: '#f2a65a',
    font: '600 15px/1 Inter, system-ui, sans-serif', opacity: '0', transition: 'opacity .25s', pointerEvents: 'none' });
  top(cap);
  top(fast);
  top(cur);
  document.addEventListener('mousemove', (e) => (cur.style.transform = `translate(${e.clientX - 4}px, ${e.clientY - 2}px)`), true);
  document.addEventListener('mousedown', () => ring.animate([{ opacity: 1, transform: 'scale(.4)' }, { opacity: 0, transform: 'scale(1.6)' }],
    { duration: 500, easing: 'ease-out' }), true);
  window.__demo = {
    say(text) { cap.style.opacity = '0'; setTimeout(() => { cap.textContent = text; cap.style.opacity = text ? '1' : '0'; }, text && cap.textContent ? 300 : 0); },
    raise() { for (const el of [cap, fast, cur]) { el.hidePopover(); el.showPopover(); } },
    fast(on) { fast.style.opacity = on ? '1' : '0'; },
  };
}"""

SCROLL = """async ([y, ms]) => {
  const y0 = scrollY, t0 = performance.now();
  const ease = (t) => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  await new Promise((done) => { const f = (now) => { const t = Math.min(1, (now - t0) / ms);
    window.scrollTo({ top: y0 + (y - y0) * ease(t), behavior: 'instant' }); t < 1 ? requestAnimationFrame(f) : done(); }; requestAnimationFrame(f); });
}"""


class Recorder:
    def __init__(self, page: Page):
        self.page = page
        self.frames: list[tuple[float, bytes]] = []
        self.warps: list[list[float]] = []  # [start, end] of each sped-up stretch, wall-clock seconds
        self.x, self.y = W * 0.75, H * 0.6

    async def start(self):
        self.cdp = await self.page.context.new_cdp_session(self.page)

        async def frame(p):
            # stamped on arrival, on the same clock as the sped-up stretches
            self.frames.append((time.time(), base64.b64decode(p["data"])))
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})

        self.cdp.on("Page.screencastFrame", lambda p: asyncio.ensure_future(frame(p)))
        await self.cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 88,
                "maxWidth": round(W * SCALE),
                "maxHeight": round(H * SCALE),
                "everyNthFrame": 2,
            },
        )
        self.t0 = time.time()

    async def stop(self):
        await self.cdp.send("Page.stopScreencast")

    async def fast(self, on: bool):
        """Start or end a stretch that plays at FAST x speed (the badge says so)."""
        await self.page.evaluate("(on) => window.__demo.fast(on)", on)
        if on:
            await asyncio.sleep(0.3)  # the badge is up before the speed changes
            self.warps.append([time.time(), float("inf")])
        else:
            self.warps[-1][1] = time.time()
            await asyncio.sleep(0.3)

    # ---------------------------------------------------------------- acting
    async def say(self, text: str, hold: float = 0):
        await self.page.evaluate("(t) => window.__demo.say(t)", text)
        await asyncio.sleep(hold)

    async def move(self, selector: str, dur: float = 0.9, dx: float = 0.5, dy: float = 0.5):
        box = await self.page.locator(selector).first.bounding_box()
        if (
            box["y"] < 60 or box["y"] + box["height"] > H - 130
        ):  # off screen or under the caption: bring it up first
            y = await self.page.evaluate("scrollY") + box["y"] + box["height"] / 2 - H * 0.45
            await self.page.evaluate(SCROLL, [y, 700])
            box = await self.page.locator(selector).first.bounding_box()
        tx, ty = box["x"] + box["width"] * dx, box["y"] + box["height"] * dy
        n = max(2, int(dur * 60))
        x0, y0 = self.x, self.y
        for i in range(1, n + 1):
            t = i / n
            e = t * t * (3 - 2 * t)
            await self.page.mouse.move(x0 + (tx - x0) * e, y0 + (ty - y0) * e)
            await asyncio.sleep(dur / n)
        self.x, self.y = tx, ty

    async def click(self, selector: str, dur: float = 0.9, pause: float = 0.25):
        await self.move(selector, dur)
        await asyncio.sleep(pause)
        await self.page.mouse.down()
        await asyncio.sleep(0.08)
        await self.page.mouse.up()

    async def scroll_to(self, selector: str, offset: int = 40, ms: int = 1300):
        y = await self.page.evaluate(
            "([s, o]) => document.querySelector(s).getBoundingClientRect().top + scrollY - o",
            [selector, offset],
        )
        await self.page.evaluate(SCROLL, [y, ms])
        await asyncio.sleep(0.2)

    async def until(self, expr: str, timeout: float = 300):
        end = time.monotonic() + timeout
        while not await self.page.evaluate(expr):
            if time.monotonic() > end:
                raise TimeoutError(expr)
            await asyncio.sleep(0.2)


async def act(r: Recorder) -> float:
    """The storyboard. Returns the time (since the start) of the frame used as the poster."""
    p = r.page
    await r.say("HumanDiT invents faces of people who don't exist.", 2.6)
    await r.move(".mosaic-grid", 1.4, 0.55, 0.45)
    await r.say("It runs on your own graphics chip, right in the browser. Nothing is uploaded.", 2.8)

    # live mode: download once, prepare, time this device
    await r.click(".gpu-strip-go", 1.1)
    await asyncio.sleep(0.6)
    await p.evaluate("window.__demo.raise()")
    await r.say("Switch it on: the page names your graphics chip and shows how fast others run.", 3.0)
    await r.click(".live-go", 1.0)
    await r.say("The AI downloads once (553 MB) and stays in your browser for next time.", 0.5)
    await r.until("document.querySelector('.live-steps li[data-step=prepare]').dataset.state !== 'todo'", 600)
    await r.say("Then it's prepared for this graphics card and timed with three faces.", 0.5)
    await r.fast(True)
    await r.until("document.querySelector('.live-go span').textContent === 'Start making faces'", 600)
    await r.fast(False)
    speed = await p.inner_text(".live-steps li[data-step=test] .ls-info")
    normal = next(
        d["instant_s"] for d in json.loads((OUT / "speeds.json").read_text())["devices"] if d["kind"] == "gpu"
    )
    # the screen recording shares the graphics card, so the speed measured during it is slower than normal: say both
    await r.say(
        f"Running: {speed} while recording the screen ({normal:.2f} s without) on this laptop's GTX 1650 Ti.",
        3.6,
    )
    await r.click(".live-go", 0.9)
    await asyncio.sleep(1.6)

    # make faces with chosen features
    await r.scroll_to(".studio-grid", 70, 900)
    await r.say("Choose features. Here: glasses and a smile.", 0.6)
    await r.click('.chip[data-feature="Glasses"]', 1.0)
    await asyncio.sleep(0.4)
    await r.click('.chip[data-feature="Smile"]', 0.7)
    await asyncio.sleep(0.5)
    await r.click(".btn-make", 0.9)
    await r.until("document.querySelector('.results-tag').classList.contains('live')", 300)
    await r.say("A second AI checks every face before you see it, and redraws any that miss.", 0.8)
    poster = time.time() - r.t0 + 1.2
    await r.move(".faces .face:nth-child(2) .face-cap", 1.2)
    await asyncio.sleep(2.2)

    # change one thing
    await r.scroll_to("#edit", 30, 1200)
    await r.say("Change one thing: the same invented person, with and without a hat.", 0.4)
    await r.click('.edit-pills .pill:text-is("Hat")', 1.0)
    await r.until("/^Live/.test(document.querySelector('.edit-status').textContent)", 300)
    await asyncio.sleep(0.6)
    box = await p.locator(".compare").bounding_box()
    await r.move(".compare", 0.8, 0.5, 0.55)
    await p.mouse.down()
    for x in (0.5, 0.12, 0.88, 0.5):
        await r.move(".compare", 0.9, x, 0.55)
    await p.mouse.up()
    _ = box
    await asyncio.sleep(1.2)

    # instant or best, raced on this GPU
    await r.scroll_to("#speed", 30, 1300)
    await r.say("One step or twenty-five? Raced live on this graphics card.", 0.4)
    await r.click(".btn-race", 1.0)
    await asyncio.sleep(0.8)
    await r.fast(True)  # the first race downloads and prepares the Best quality model (once)
    await r.until("/^Live/.test(document.querySelector('.race-hw').textContent)", 600)
    await r.fast(False)
    await r.until("document.querySelectorAll('.lane.done').length === 4", 600)
    await asyncio.sleep(1.0)

    # the watermark
    await r.scroll_to("#check", 30, 1300)
    await r.say("Every face carries an invisible watermark. The check runs on your device.", 0.4)
    await r.click(".example", 1.0)
    await r.until("document.querySelector('.verdict').classList.contains('yes')", 300)
    await asyncio.sleep(2.4)

    await r.scroll_to("#top", 0, 1600)
    await r.say("No account, nothing uploaded. Try it yourself.", 3.5)
    return poster


def encode(
    frames: list[tuple[float, bytes]], t0: float, out: Path, poster_t: float, warps: list[list[float]]
) -> float:
    """Lay the screencast's frames (sent when something changes) onto a steady 30 fps timeline and encode it."""
    frames.sort(key=lambda f: f[0])
    start, end = frames[0][0], frames[-1][0] + 1.0
    times = []  # the source moment shown by each output frame
    t = start
    while t < end:
        times.append(t)
        t += (FAST if any(a <= t < b for a, b in warps) else 1) / FPS
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
           "-framerate", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow", "-crf", "22",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]  # fmt: skip
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    k = 0
    poster = None
    for t in times:
        while k + 1 < len(frames) and frames[k + 1][0] <= t:
            k += 1
        ff.stdin.write(frames[k][1])
        if poster is None and t - t0 >= poster_t:
            poster = frames[k][1]
    ff.stdin.close()
    ff.wait()
    (out.parent / "demo_poster.jpg").write_bytes(poster or frames[len(frames) // 3][1])
    return len(times) / FPS


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    a = ap.parse_args()
    async with async_playwright() as pw:
        b = await pw.chromium.launch(
            channel="msedge", headless=True, args=["--enable-unsafe-webgpu", "--force_high_performance_gpu"]
        )
        ctx = await b.new_context(viewport={"width": W, "height": H}, device_scale_factor=SCALE)
        page = await ctx.new_page()
        await page.goto(a.url, wait_until="load")
        await page.wait_for_timeout(2500)
        await page.evaluate(OVERLAY)
        await page.mouse.move(W * 0.75, H * 0.6)
        r = Recorder(page)
        await r.start()
        poster = await act(r)
        await asyncio.sleep(0.5)
        await r.stop()
        await b.close()
    secs = encode(r.frames, r.t0, OUT / "demo.mp4", poster, r.warps)
    mb = (OUT / "demo.mp4").stat().st_size / 2**20
    print(f"{OUT / 'demo.mp4'}: {secs:.0f} s, {len(r.frames)} frames captured, {mb:.1f} MB")


if __name__ == "__main__":
    asyncio.run(main())
