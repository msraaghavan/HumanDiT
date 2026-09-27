"""Measure how smoothly the demo page animates: frame times in a real browser while the motion runs.

    uv run python scripts/motion_fps.py [--url http://localhost:7860] [--headed]

For each scene (the opening, the how-it-works stage, the cursor moving over the wall of faces, a scroll through the
page) it records every requestAnimationFrame interval for a few seconds and reports the median frame time, the mean
frame rate and the share of frames slower than 25 ms (visible stutter at 60 Hz). Headless browsers render on the CPU
unless --headed is given, so compare runs made the same way.
"""

from __future__ import annotations

import argparse
import json
import statistics

from playwright.sync_api import Page, sync_playwright

PROBE = """async ms => {
  const t = []; let last = performance.now();
  await new Promise(done => {
    const end = last + ms;
    const tick = now => { t.push(now - last); last = now; now < end ? requestAnimationFrame(tick) : done(); };
    requestAnimationFrame(tick);
  });
  return t.slice(2);
}"""


START = "ms => { window.__hdProbe = (" + PROBE + ")(ms); }"  # measure in the background while Playwright acts


def measure(page: Page, ms: int = 4000) -> dict:
    return summary(page.evaluate(PROBE, ms))


def summary(dt: list[float]) -> dict:
    return {
        "frames": len(dt),
        "fps": round(1000 * len(dt) / sum(dt), 1),
        "median_ms": round(statistics.median(dt), 1),
        "slow_%": round(100 * sum(d > 25 for d in dt) / len(dt), 1),
        "worst_ms": round(max(dt), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel, headless=not a.headed)
        page = b.new_page(viewport={"width": 1280, "height": 900})
        page.goto(a.url, wait_until="load")
        page.locator(".hd-hero").wait_for(timeout=120_000)
        page.get_by_text("Made 4 faces").wait_for(timeout=180_000)  # the page's first job has finished
        page.wait_for_timeout(1500)
        out = {"opening (hero + reel + wall)": measure(page)}
        page.locator(".hd-how").scroll_into_view_if_needed()
        out["how-it-works stage"] = measure(page)
        page.evaluate("window.scrollTo(0, 0)")
        box = page.locator(".hd-hero").bounding_box()
        page.mouse.move(box["x"] + 100, box["y"] + 100)
        page.evaluate(START, 3000)  # measures while the cursor sweeps
        for i in range(60):
            page.mouse.move(box["x"] + 100 + i * 17, box["y"] + 100 + (i % 20) * 12)
            page.wait_for_timeout(40)
        out["cursor over the wall"] = summary(page.evaluate("() => window.__hdProbe"))
        page.mouse.move(5, 895)
        page.evaluate(START, 3000)
        for _ in range(30):
            page.mouse.wheel(0, 90)
            page.wait_for_timeout(60)
        out["scrolling"] = summary(page.evaluate("() => window.__hdProbe"))
        b.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
