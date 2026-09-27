"""Drive the running Gradio demo in a real browser and save README screenshots.

    uv run python app/gradio_app.py ...          # in another terminal (http://localhost:7860)
    uv run python scripts/demo_screenshots.py    # uses the installed Microsoft Edge (channel msedge)

Writes docs/assets/demo_hero.jpg (the page's opening, with the cursor lighting the wall of faces), demo_generate.jpg,
demo_edit.jpg, demo_compare.jpg, demo_trajectory.jpg, demo_watermark.jpg, demo_scoreboard.jpg and demo_mobile.jpg.
Faces are made with a pinned seed (the engineers' drawer), so the shots are reproducible. Each shot waits for its
animation to finish; the page's intro curtain is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import Locator, Page, sync_playwright

SKIP_INTRO = "try { sessionStorage.setItem('hd-intro', '1') } catch (e) {}"
FREEZE_REEL = (
    "t => { const v = document.querySelector('.hd-hero video'); if (v) { v.pause(); v.currentTime = t; } }"
)


def open_tab(page: Page, name: str, top: float) -> Locator:
    """Open a tab with the page scrolled to the tabs: the opening's animations then pause, and on a laptop they would
    otherwise compete with the model for the machine (the live timings in the shots are the model's alone)."""
    page.get_by_role("tab", name=name).click()
    page.evaluate("y => window.scrollTo(0, y)", top)
    return page.locator("[role=tabpanel]:visible").last


def settle(page: Page) -> None:
    """Wait until no Gradio job is running (its "processing | 1.2/3.4s" overlays are gone)."""
    page.wait_for_timeout(800)
    page.get_by_text("processing |").first.wait_for(state="hidden", timeout=300_000)


def shot(page: Page, path: Path, top: float, bottom_of: Locator, pad: int = 16, wait: int = 1200) -> None:
    page.wait_for_timeout(wait)
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
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 1000}, device_scale_factor=1, color_scheme="dark"
        )
        ctx.add_init_script(SKIP_INTRO)
        page = ctx.new_page()
        page.goto(a.url, wait_until="load")  # the looping showreel never lets the network idle
        page.get_by_text("Made 4 faces").wait_for(timeout=180_000)  # the page makes faces on load

        # the opening: headline, measured numbers, the wall of faces lit by the cursor, the how-it-works stage
        page.evaluate(FREEZE_REEL, 6.8)
        hero = page.locator(".hd-hero").bounding_box()
        page.mouse.move(
            hero["x"] + hero["width"] * 0.92, hero["y"] + hero["height"] * 0.95
        )  # the torch, clear of text
        page.wait_for_function(  # the stage's 4th card, one feature in: "+ Smile"
            "() => [...document.querySelectorAll('.hd-step')].findIndex(s => s.classList.contains('on')) === 3",
            timeout=30_000,
        )
        page.wait_for_timeout(2900)
        shot(page, out / "demo_hero.jpg", 0, page.locator(".hd-how"), pad=8, wait=0)
        page.mouse.move(5, 995)
        tabs_top = (
            page.get_by_role("tablist").first.bounding_box()["y"] + page.evaluate("window.scrollY") - 10
        )

        # Make faces: 8 faces, must have glasses, blond hair and a smile, must not be a man; seed 42
        page.evaluate("y => window.scrollTo(0, y)", tabs_top)
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
            page.get_by_role("button", name="Make faces").click()
            settle(page)
        panel = page.locator("[role=tabpanel]:visible").last
        shot(page, out / "demo_generate.jpg", tabs_top, panel.get_by_text("Made 8 faces"), pad=40, wait=2600)

        # Change one thing: fills itself when opened; caught mid-wipe
        panel = open_tab(page, "Change one thing", tabs_top)
        panel.locator(".hd-wipe").wait_for(timeout=180_000)
        settle(page)
        page.wait_for_timeout(1650)
        shot(page, out / "demo_edit.jpg", tabs_top, panel.locator("img").nth(4), wait=0)

        # Instant or best?: the one-step student vs the teacher at 10 / 25 / 50 steps, same start; the race replays
        panel = open_tab(page, "Instant or best?", tabs_top)
        panel.locator(".hd-race").wait_for(timeout=300_000)
        settle(page)
        panel.get_by_role("button", name="Compare", exact=True).click()  # again, warm: steady-state timings
        settle(page)
        shot(page, out / "demo_compare.jpg", tabs_top, panel.locator(".hd-race"), pad=24, wait=7000)

        # Watch it draw: the teacher's drafts, the video's step counter
        panel = open_tab(page, "Watch it draw", tabs_top)
        panel.get_by_text("Step 25 of 25").wait_for(timeout=300_000)
        shot(
            page, out / "demo_trajectory.jpg", tabs_top, panel.get_by_text("Step 25 of 25"), pad=30, wait=2000
        )

        # Is it AI-made?: the watermark check on the bundled HumanDiT example, after the scan and the stamp
        panel = open_tab(page, "Is it AI-made?", tabs_top)
        panel.locator("button:has(img)").first.click()
        page.get_by_text("carries the HumanDiT watermark").wait_for(timeout=120_000)
        shot(
            page,
            out / "demo_watermark.jpg",
            tabs_top,
            panel.locator("button:has(img)").first,
            pad=30,
            wait=2200,
        )

        # For engineers: every model's measured realism gap
        open_tab(page, "For engineers", tabs_top)
        board = page.locator(".hd-score")
        board.scroll_into_view_if_needed()
        page.wait_for_timeout(3000)
        board.screenshot(path=str(out / "demo_scoreboard.jpg"), quality=90)
        ctx.close()

        # a phone: the same opening, one column
        ctx = browser.new_context(
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
            is_mobile=True,
            has_touch=True,
            color_scheme="dark",
        )
        ctx.add_init_script(SKIP_INTRO)
        page = ctx.new_page()
        page.goto(a.url, wait_until="load")  # the looping showreel never lets the network idle
        page.locator(".hd-hero.hd-go").wait_for(timeout=60_000)
        page.evaluate(FREEZE_REEL, 6.8)
        page.wait_for_timeout(4000)
        page.screenshot(path=str(out / "demo_mobile.jpg"), quality=85)
        browser.close()
    print(f"saved screenshots to {out}")


if __name__ == "__main__":
    main()
