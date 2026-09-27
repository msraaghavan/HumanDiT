"""Drive the running Gradio demo in a real browser and save README screenshots.

    uv run python app/gradio_app.py ...          # in another terminal (http://localhost:7860)
    uv run python scripts/demo_screenshots.py    # uses the installed Microsoft Edge (channel msedge)

Writes docs/assets/demo_generate.jpg (with the page header), demo_edit.jpg, demo_compare.jpg, demo_trajectory.jpg and
demo_watermark.jpg. Faces are made with a pinned seed (the engineers' drawer), so the shots are reproducible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright


def open_tab(page: Page, name: str) -> Locator:
    page.get_by_role("tab", name=name).click()
    return page.locator("[role=tabpanel]:visible").last


def settle(page: Page) -> None:
    """Wait until no Gradio job is running (its "processing | 1.2/3.4s" overlays are gone)."""
    page.wait_for_timeout(800)
    page.get_by_text("processing |").first.wait_for(state="hidden", timeout=300_000)


def shot(page: Page, path: Path, top: float, bottom_of: Locator, pad: int = 16) -> None:
    page.wait_for_timeout(1200)
    box = bottom_of.bounding_box()  # viewport coordinates; the clip below is in page coordinates
    box["y"] += page.evaluate("window.scrollY")
    page.screenshot(
        path=str(path),
        full_page=True,
        clip={"x": 0, "y": top, "width": 1280, "height": box["y"] + box["height"] + pad - top},
        quality=90,
    )


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
        page = browser.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1)
        page.goto(a.url, wait_until="networkidle")
        page.get_by_text("Made 4 faces").wait_for(timeout=180_000)  # the page makes faces on load (warm-up)
        # freeze the header reel on one frame
        page.evaluate(
            "() => { const v = document.querySelector('#reel video'); if (v) { v.pause(); v.currentTime = 6.8; } }"
        )
        tabs_top = (
            page.get_by_role("tablist").first.bounding_box()["y"] + page.evaluate("window.scrollY") - 10
        )

        # Make faces: 8 faces, must have glasses, blond hair and a smile, must not be a man; seed 42
        page.get_by_text("For engineers: sampler settings").click()
        page.get_by_label("Seed (the noise the faces start from)").fill("42")
        page.get_by_label("Keep the same people (reuse this seed)").check()
        page.get_by_label("How many faces?").last.fill("8")
        page.keyboard.press("Enter")
        must, never = (
            page.locator(".block", has_text="Must have"),
            page.locator(".block", has_text="Must NOT have"),
        )
        for name in ("Glasses", "Blond hair", "Smile"):
            must.last.get_by_label(name, exact=True).check()
        never.last.get_by_label("Man", exact=True).check()
        page.get_by_text("For engineers: sampler settings").click()  # close the drawer again
        # the first batch of 8 includes CUDA warm-up for the new batch size: report the second
        for _ in range(2):
            page.get_by_role("button", name="✨ Make faces").click()
            settle(page)
        panel = page.locator("[role=tabpanel]:visible").last
        shot(page, out / "demo_generate.jpg", 0, panel.get_by_text("Made 8 faces"), pad=40)

        # Change one thing: fills itself when opened (same person without / AI's choice / with glasses)
        panel = open_tab(page, "✏️ Change one thing")
        panel.locator("img").nth(2).wait_for(timeout=180_000)
        shot(page, out / "demo_edit.jpg", tabs_top, panel.locator("img").nth(2))

        # Instant or best?: the one-step student vs the teacher at 10 / 25 / 50 steps, same start
        panel = open_tab(page, "⚡ Instant or best?")
        panel.get_by_text("Best quality · 50 steps").wait_for(timeout=300_000)
        settle(page)
        panel.get_by_role("button", name="Compare", exact=True).click()  # again, warm: steady-state timings
        settle(page)
        shot(page, out / "demo_compare.jpg", tabs_top, panel.get_by_text("Realism gap").last, pad=24)

        # Watch it draw: the teacher's drafts
        panel = open_tab(page, "🔍 Watch it draw")
        panel.get_by_text("Step 25 of 25").wait_for(timeout=300_000)
        shot(page, out / "demo_trajectory.jpg", tabs_top, panel.get_by_text("Step 25 of 25"), pad=30)

        # Is it AI-made?: the watermark check on the bundled HumanDiT example
        panel = open_tab(page, "🛡️ Is it AI-made?")
        panel.locator("button:has(img)").first.click()
        page.get_by_text("carries the HumanDiT watermark").wait_for(timeout=120_000)
        shot(page, out / "demo_watermark.jpg", tabs_top, panel.locator("button:has(img)").first, pad=30)
        browser.close()
    print(f"saved screenshots to {out}")


if __name__ == "__main__":
    main()
