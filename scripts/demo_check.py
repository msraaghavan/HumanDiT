"""End-to-end check of the running demo in a real browser: every tab works, and the page is what it should be.

    uv run python scripts/demo_check.py [--url http://localhost:7860]

Checks: no JavaScript errors; dark mode; no Gradio footer; no emoji anywhere in the page text; the opening, the
how-it-works stage and the reel load; and each tool returns its result (faces with the checker's marks, the edit and
its wipe, the blend video, the drawing video and drafts, the race, the watermark verdict for both examples, the
engineers' scoreboard). Prints one line per check and exits non-zero if any fails.
"""

from __future__ import annotations

import argparse
import re
import sys

from playwright.sync_api import Page, sync_playwright

EMOJI = re.compile("[\U0001f000-\U0001faff☀-⛿✀-✒✔-✖✘-➿⬀-⯿️]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    ap.add_argument("--channel", default="msedge")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # captions carry ✓ / ✗
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}", flush=True)

    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel)
        page: Page = b.new_page(viewport={"width": 1280, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(a.url, wait_until="load")
        page.get_by_text("Made 4 faces").wait_for(timeout=180_000)
        page.wait_for_timeout(2500)

        check("dark mode", page.evaluate("document.body.classList.contains('dark')"))
        check("no Gradio footer", page.locator("footer:visible").count() == 0)
        check("hero animated in", page.locator(".hd-hero.hd-go").count() == 1)
        check(
            "reel playing",
            page.evaluate(
                "(() => { const v = document.querySelector('.hd-hero video'); return !!v && !v.paused && v.readyState >= 2 })()"
            ),
        )
        check(
            "wall of faces drawn", page.evaluate("document.querySelectorAll('.hd-hero canvas').length") == 2
        )
        check("how-it-works stage running", page.locator(".hd-step.on").count() == 1)
        caps = page.locator(".hd-dev .caption-label").all_inner_texts()
        check("make faces: 4 faces with the checker's words", len(caps) == 4, "; ".join(caps)[:120])

        # make faces with a request: the marks must follow what was asked
        page.locator(".block", has_text="Must have").last.get_by_label("Glasses", exact=True).check()
        page.get_by_role("button", name="Make faces").click()
        page.wait_for_function(  # the new faces: captions now judge the requested feature
            "() => [...document.querySelectorAll('.hd-dev .caption-label')].every(c => /[✓✗] glasses/.test(c.textContent))",
            timeout=120_000,
        )
        caps = page.locator(".hd-dev .caption-label").all_inner_texts()
        check(
            "make faces with 'Glasses': every caption judges glasses",
            all("glasses" in c for c in caps),
            caps[0] if caps else "",
        )
        page.locator(".block", has_text="Must have").last.get_by_label("Glasses", exact=True).uncheck()

        def tab(name: str) -> None:
            page.get_by_role("tab", name=name).click()

        tab("Change one thing")
        page.locator(".hd-wipe").wait_for(timeout=180_000)
        check(
            "change one thing: wipe + three pictures",
            page.locator(".hd-wipe img").count() == 2
            and page.locator("[role=tabpanel]:visible img").count() >= 5,
        )

        tab("Blend two faces")
        page.wait_for_function(
            "() => { const v = document.querySelector('#blend-video video'); return v && v.duration > 0 }",
            timeout=180_000,
        )
        page.wait_for_timeout(600)
        check(
            "blend: video plays with its A-to-B bar", page.locator("#blend-video .hd-blendbar").count() == 1
        )

        tab("Watch it draw")
        page.get_by_text("Step 25 of 25").wait_for(timeout=180_000)
        page.wait_for_timeout(600)
        check(
            "watch it draw: video, step counter and six drafts",
            page.locator("#draw-video .hd-ticks").count() == 1
            and page.locator(".hd-dev .thumbnail-item").count() >= 6,
        )

        tab("Instant or best?")
        page.locator(".hd-race").wait_for(timeout=300_000)
        lanes = page.locator(".hd-lane.time").count()
        check(
            "instant or best: 4 lanes timed live + realism gaps",
            lanes == 4 and page.locator(".hd-lane.gapl").count() == 4,
        )

        tab("Is it AI-made?")
        ex = page.locator("[role=tabpanel]:visible button:has(img)")
        ex.nth(0).click()
        page.get_by_text("carries the HumanDiT watermark").wait_for(timeout=120_000)
        page.wait_for_timeout(1500)
        check(
            "watermark found on a HumanDiT face",
            "hd-yes" in (page.locator("#verdict").get_attribute("class") or ""),
        )
        ex.nth(1).click()
        page.get_by_text("No HumanDiT watermark found").wait_for(timeout=120_000)
        page.wait_for_timeout(1500)
        check(
            "no watermark on the first version's face",
            "hd-no" in (page.locator("#verdict").get_attribute("class") or ""),
        )

        tab("For engineers")
        page.locator(".hd-score").scroll_into_view_if_needed()
        check("engineers: scoreboard of 8 measured models", page.locator(".hd-srow").count() == 8)

        text = page.evaluate("document.body.innerText")
        found = sorted(set(EMOJI.findall(text)))
        check("no emoji in the page", not found, " ".join(found))
        real = [e for e in errors if "favicon" not in e]
        check("no JavaScript errors", not real, " | ".join(real)[:200])
        b.close()
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
