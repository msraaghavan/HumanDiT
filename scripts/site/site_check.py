"""End-to-end check of the website (site/) in a real browser, against a running engine, plus its frame rate.

    uv run --extra demo python app/api.py --port 7861 &          # the engine
    uv run python scripts/site/serve.py --port 8081 &             # the page, under the production CSP
    uv run python scripts/site/site_check.py [--url http://localhost:8081/] [--fps]

Checks: no JavaScript errors; the engine connects; every section starts from its saved results; each live tool
answers (faces that honour a ticked feature, another person, a new blend, a live race, both watermark verdicts); no
emoji; no sideways scrolling on a phone. --fps also measures every requestAnimationFrame interval for a few seconds
per scene (the hero, scrubbing how-it-works, the blend playing, scrolling). Exits non-zero if a check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time

from playwright.sync_api import Page, sync_playwright

EMOJI = re.compile("[\U0001f000-\U0001faff☀-⛿✀-➿⬀-⯿️]")
FPS_JS = """async (ms) => { const d = []; let last = performance.now();
  await new Promise(res => { const t0 = last; const f = now => { d.push(now - last); last = now;
    if (now - t0 < ms) requestAnimationFrame(f); else res(); }; requestAnimationFrame(f); });
  d.shift(); d.sort((a, b) => a - b);
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  return { fps: 1000 / mean, p95: d[Math.floor(d.length * 0.95)], slow: d.filter(x => x > 25).length / d.length }; }"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--fps", action="store_true")
    ap.add_argument("--headed", action="store_true", help="a real window (GPU compositing) for --fps")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  (' + detail + ')' if detail else ''}", flush=True)

    def wait_js(pg: Page, expr: str, timeout: int) -> None:
        """Poll a JavaScript expression (wait_for_function compiles it with eval, which the site's CSP blocks)."""
        end = time.monotonic() + timeout / 1000
        while not pg.evaluate(expr):
            if time.monotonic() > end:
                raise TimeoutError(expr)
            pg.wait_for_timeout(250)

    with sync_playwright() as p:
        b = p.chromium.launch(channel=a.channel, headless=not a.headed)
        page: Page = b.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(a.url, wait_until="load")
        wait_js(page, "document.querySelector('.engine').dataset.state === 'ready'", timeout=240_000)
        check("engine connects", True, page.inner_text(".engine-text"))
        page.wait_for_timeout(1500)

        fps = {}
        if a.fps:
            fps["hero"] = page.evaluate(FPS_JS, 3000)

        # ------------------------------------------------------------ saved results everywhere
        blank = """(sel) => { const c = document.querySelector(sel); const x = c.getContext('2d')
            .getImageData(c.width / 2 - 8, c.height / 2 - 8, 16, 16).data; let s = 0;
            for (let i = 0; i < x.length; i += 4) s += x[i] + x[i + 1] + x[i + 2]; return s > 0 }"""
        check("hero: portrait is drawing", page.evaluate(blank, ".portrait-canvas"))
        check("hero: wall of faces", page.locator(".wall-col").count() >= 5)
        check(
            "hero: 4 measured numbers",
            page.locator(".stat").count() == 4,
            page.inner_text(".stats").replace("\n", " ")[:120],
        )
        page.evaluate("document.querySelector('.how').scrollIntoView()")
        top, h = page.evaluate(
            "[document.querySelector('.how').offsetTop, document.querySelector('.how').offsetHeight]"
        )
        page.evaluate(f"window.scrollTo(0, {top} + ({h} - innerHeight) * 0.4)")
        page.wait_for_timeout(500)
        step = page.inner_text(".how-count b")
        check(
            "how it works: scrolling moves the drawing",
            step not in ("00", "01"),
            f"step {step}, {page.inner_text('.how-left')}",
        )
        if a.fps:
            page.evaluate(f"window.scrollTo(0, {top})")
            fps["how (scrubbing)"] = page.evaluate(
                "async (a) => { const run = (%s); const go = (async () => { for (let i = 0; i < 120; i++) { "
                "window.scrollBy(0, a); await new Promise(r => setTimeout(r, 25)); } })(); const r = await run(3000); "
                "await go; return r; }".replace("%s", FPS_JS),
                max(4, (h - 900) // 120),
            )
        page.locator("#studio").scroll_into_view_if_needed()
        page.wait_for_timeout(1200)
        check(
            "make faces: 8 saved faces",
            page.locator(".face img").count() == 8,
            page.inner_text(".results-tag"),
        )

        # ------------------------------------------------------------ live: make faces with a feature
        page.locator(".chip", has_text="Glasses").click()
        page.locator('.seg[data-name="count"] button[data-value="4"]').click()
        page.locator(".btn-make").click()
        wait_js(page, "document.querySelector('.results-tag').classList.contains('live')", timeout=180_000)
        page.wait_for_timeout(1500)
        caps = page.locator(".face-cap").all_inner_texts()
        check(
            "make faces live: 4 faces, each judged on glasses",
            len(caps) == 4 and all("glasses" in c for c in caps),
            f"{page.inner_text('.results-status')} | {caps[0] if caps else ''}",
        )
        check("make faces live: each face downloadable", page.locator(".face-dl").count() == 4)

        # ------------------------------------------------------------ live: change one thing
        page.locator("#edit").scroll_into_view_if_needed()
        page.locator(".btn-edit").click()
        wait_js(
            page, "document.querySelector('.edit-status').textContent.startsWith('Live')", timeout=180_000
        )
        check("change one thing: another person, live", True, page.inner_text(".edit-status"))

        # ------------------------------------------------------------ live: blend
        page.locator("#blend").scroll_into_view_if_needed()
        page.wait_for_timeout(800)
        if a.fps:
            fps["blend (playing)"] = page.evaluate(FPS_JS, 3000)
        page.locator(".btn-blend").click()
        wait_js(
            page, "document.querySelector('.blend-status').textContent.startsWith('Live')", timeout=240_000
        )
        check("blend: two new people, live", True, page.inner_text(".blend-status"))

        # ------------------------------------------------------------ live: race
        page.locator("#speed").scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.locator(".btn-race").click()
        wait_js(page, "document.querySelector('.race-hw').textContent.startsWith('Live')", timeout=240_000)
        # the replay runs at each lane's real speed, so wait as long as the slowest lane really took
        with contextlib.suppress(TimeoutError):
            wait_js(page, "document.querySelectorAll('.lane.done').length === 4", timeout=60_000)
        check(
            "instant or best: 4 lanes raced live",
            page.locator(".lane.done").count() == 4,
            page.inner_text(".race-hw"),
        )

        # ------------------------------------------------------------ live: watermark
        page.locator("#check").scroll_into_view_if_needed()
        ex = page.locator(".example")
        ex.nth(0).click()
        wait_js(
            page,
            "document.querySelector('.verdict').classList.contains('yes') || document.querySelector('.verdict').classList.contains('no')",
            timeout=120_000,
        )
        check("watermark found on a HumanDiT face", "yes" in (page.get_attribute(".verdict", "class") or ""))
        ex.nth(1).click()
        wait_js(page, "document.querySelector('.verdict').classList.contains('no')", timeout=120_000)
        check("no watermark on the first version's face", True)

        page.locator("#numbers").scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        check("the numbers: 8 measured models", page.locator(".srow").count() == 8)
        if a.fps:
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(600)
            fps["scrolling"] = page.evaluate(
                "async () => { const run = (%s); const go = (async () => { for (let i = 0; i < 100; i++) { "
                "window.scrollBy(0, 90); await new Promise(r => setTimeout(r, 30)); } })(); const r = await run(3000); "
                "await go; return r; }".replace("%s", FPS_JS)
            )
        text = page.evaluate("document.body.innerText")
        found = sorted(set(EMOJI.findall(text)))
        check("no emoji", not found, " ".join(found))

        # ------------------------------------------------------------ phone
        phone = b.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        phone.goto(a.url, wait_until="load")
        phone.wait_for_timeout(3000)
        over = phone.evaluate("document.documentElement.scrollWidth - innerWidth")
        check("phone: no sideways scrolling", over <= 0, f"{over}px")
        phone.close()

        real = [e for e in errors if "favicon" not in e]
        check("no JavaScript errors", not real, " | ".join(real)[:300])
        b.close()

    for scene, r in fps.items():
        print(
            f"FPS  {scene:18s} {r['fps']:6.1f} fps  p95 {r['p95']:5.1f} ms  frames > 25 ms: {100 * r['slow']:.1f}%"
        )
    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
