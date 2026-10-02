"""The link-preview picture (site/assets/og.jpg, 1200x630): the site's own opening, rendered in Edge.

uv run python scripts/site/serve.py --port 8081 &
uv run python scripts/site/og_image.py [--url http://localhost:8081/]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[2] / "site" / "assets" / "og.jpg"


def wait_js(pg: Page, expr: str, timeout: int = 30_000) -> None:
    """Poll a JavaScript expression (wait_for_function compiles it with eval, which the site's CSP blocks)."""
    end = time.monotonic() + timeout / 1000
    while not pg.evaluate(expr):
        if time.monotonic() > end:
            raise TimeoutError(expr)
        pg.wait_for_timeout(250)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    a = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", args=["--force_high_performance_gpu"])  # no built-in-graphics note
        page = b.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        page.add_init_script("sessionStorage.setItem('hd-intro', '1')")  # no intro in the picture
        page.goto(a.url, wait_until="load")
        wait_js(
            page,
            "document.querySelector('.stage-next').classList.contains('timing')",  # the first face has formed
        )
        page.wait_for_timeout(1800)  # the words and the faces have landed
        page.evaluate(  # the CSP blocks injected <style>; element styles are fine
            "document.querySelectorAll('.scroll-cue, .live-pill').forEach((e) => (e.style.visibility = 'hidden'))"
        )
        page.screenshot(path=str(OUT), type="jpeg", quality=88)
        b.close()
    print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
