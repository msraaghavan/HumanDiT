"""The link-preview picture (site/assets/og.jpg, 1200x630): the site's own opening, rendered in Edge.

uv run python -m http.server 8080 --directory site &
uv run python scripts/site/og_image.py [--url http://localhost:8080/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[2] / "site" / "assets" / "og.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080/")
    a = ap.parse_args()
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        page = b.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
        page.add_init_script("sessionStorage.setItem('hd-intro', '1')")  # no intro in the picture
        page.goto(a.url, wait_until="load")
        # wait for the words to land and the portrait to finish a face
        page.wait_for_function("document.querySelector('.hud-step b')?.textContent === '25'", timeout=30_000)
        page.wait_for_timeout(900)
        page.add_style_tag(content=".scroll-cue, .nav { visibility: hidden }")
        page.screenshot(path=str(OUT), type="jpeg", quality=88)
        b.close()
    print(OUT, f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
