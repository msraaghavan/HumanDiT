"""Drive the running Gradio demo in a real browser and save README screenshots.

    uv run python app/gradio_app.py ...          # in another terminal (http://localhost:7860)
    uv run python scripts/demo_screenshots.py    # uses the installed Microsoft Edge (channel msedge)

Writes docs/assets/demo_generate.jpg, demo_edit.jpg, demo_compare.jpg, demo_trajectory.jpg.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Page, sync_playwright


def radio(page: Page, attribute: str, value: str) -> None:
    group = page.locator(".block", has=page.get_by_text(attribute, exact=True)).filter(
        has=page.locator("input[type=radio]")
    )
    group.last.get_by_label(value, exact=True).check()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--out", default="docs/assets")
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=a.channel)
        page = browser.new_page(viewport={"width": 1280, "height": 1100}, device_scale_factor=1)
        page.goto(a.url, wait_until="networkidle")

        # Warm-up: the first call loads TrustMark and the CUDA kernels; the screenshot should show steady state
        faces = page.get_by_role("spinbutton", name="number input for Faces")
        faces.fill("1")
        page.keyboard.press("Enter")
        page.get_by_role("button", name="Generate", exact=True).last.click()
        page.get_by_text("for 1").wait_for(timeout=180_000)

        # Generate: 1-step student, composed attributes, 8 faces
        faces.fill("8")
        page.keyboard.press("Enter")
        for attr, val in (("Male", "no"), ("Eyeglasses", "yes"), ("Blond Hair", "yes"), ("Smiling", "yes")):
            radio(page, attr, val)
        page.get_by_role("button", name="Generate", exact=True).last.click()
        page.get_by_text("for 8").wait_for(timeout=180_000)
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(
            path=str(out / "demo_generate.jpg"),
            clip={"x": 0, "y": 0, "width": 1280, "height": 1000},
            quality=90,
        )

        # Edit: same seed, one attribute absent / unspecified / present (1-step student, the tab's default)
        page.get_by_role("tab", name="Edit").click()
        page.get_by_role("button", name="Show absent · any · present").click()
        page.locator("[role=tabpanel]:visible img").nth(2).wait_for(timeout=180_000)
        page.wait_for_timeout(2500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(
            path=str(out / "demo_edit.jpg"), clip={"x": 0, "y": 0, "width": 1280, "height": 620}, quality=90
        )

        # Compare samplers: one-step student vs the teacher at 10 / 25 / 50 steps, same noise
        page.get_by_role("tab", name="Compare samplers").click()
        page.get_by_role("button", name="Compare", exact=True).click()
        page.get_by_text("teacher · 50 steps").wait_for(timeout=300_000)  # the caption, not the page header
        page.wait_for_timeout(2500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(
            path=str(out / "demo_compare.jpg"),
            clip={"x": 0, "y": 0, "width": 1280, "height": 730},
            quality=90,
        )

        # Under the hood: teacher trajectory (multi-step, slower on a laptop GPU)
        page.get_by_role("tab", name="Under the hood").click()
        page.get_by_role("button", name="Show trajectory").click()
        page.get_by_text("step 25/25").wait_for(timeout=300_000)
        page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(500)
        page.screenshot(
            path=str(out / "demo_trajectory.jpg"),
            clip={"x": 0, "y": 0, "width": 1280, "height": 560},
            quality=90,
        )
        browser.close()
    print(f"saved screenshots to {out}")


if __name__ == "__main__":
    main()
