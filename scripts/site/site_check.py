"""End-to-end check of the website (site/) in a real browser, with the AI running in the page, plus its frame rate.

    uv run python scripts/site/serve.py --port 8081 &             # the page and its models, under the production CSP
    uv run python scripts/site/site_check.py [--url http://localhost:8081/] [--fps] [--cpu]

Checks: no JavaScript errors and no missing files; the pixel-art town draws; every section starts from its saved
results; the AI check reads both examples on this device; the live dialog names the graphics chip and says where the
AI will be stored; live mode downloads, prepares and times itself through the page's own buttons, and a button pressed
meanwhile says to wait and then runs live; each tool answers live (faces that honour a ticked feature, a feature
Instant mode can't draw made in Best quality, another person, a race on this GPU); removing the AI empties the
browser's storage; no emoji; no sideways scrolling on a phone. --cpu runs the AI on the processor
(WebAssembly) instead of WebGPU. --fps also measures every requestAnimationFrame interval for a few seconds per scene.
Exits non-zero if a check fails.
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
# a canvas is drawing if its middle has some light in it
LIT = """(sel) => { const c = document.querySelector(sel); if (!c || !c.width) return false;
  const x = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let s = 0;
  for (let i = 0; i < x.length; i += 4) s += x[i] + x[i + 1] + x[i + 2]; return s / (x.length / 4) > 12 }"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--fps", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="run the AI on the processor, not WebGPU")
    ap.add_argument("--headed", action="store_true", help="a real window (GPU compositing) for --fps")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    url = a.url + ("?ai=cpu" if a.cpu else "")
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
        b = p.chromium.launch(
            channel=a.channel,
            headless=not a.headed,
            args=["--enable-unsafe-webgpu", "--force_high_performance_gpu"],
        )
        page: Page = b.new_page(viewport={"width": 1440, "height": 900})
        errors: list[str] = []
        missing: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("response", lambda r: missing.append(r.url) if r.status >= 400 else None)
        page.goto(url, wait_until="load")
        page.wait_for_timeout(3000)

        fps = {}
        if a.fps:
            fps["hero"] = page.evaluate(FPS_JS, 3000)

        # ------------------------------------------------------------ the page before live mode
        check("no page header: it opens on the hero", page.locator("body > header, .nav").count() == 0)
        check("hero: the pixel-art town is drawing", page.evaluate(LIT, ".hero-scene canvas"))
        size = page.evaluate(
            "(() => { const c = document.querySelector('.hero-scene canvas'); return [c.width, c.height, c.style.width] })()"
        )
        check(
            "hero: the town is scaled by a whole number",
            True,
            f"{size[0]} x {size[1]} art pixels shown at {size[2]}",
        )
        check("hero: portrait is drawing", page.evaluate(LIT, ".portrait-canvas"))
        check("hero: 4 measured numbers", page.locator(".stat").count() == 4)
        check(
            "live mode starts off: saved results",
            page.get_attribute(".live-pill", "data-state") == "saved",
            page.inner_text(".live-pill .engine-text"),
        )
        page.evaluate("document.querySelector('.interlude').scrollIntoView()")
        page.wait_for_timeout(1500)
        check("interlude: the town draws full screen", page.evaluate(LIT, ".interlude canvas"))
        if a.fps:
            fps["interlude (town)"] = page.evaluate(FPS_JS, 3000)
        page.evaluate("document.querySelector('.how').scrollIntoView()")
        top, h = page.evaluate(
            "[document.querySelector('.how').offsetTop, document.querySelector('.how').offsetHeight]"
        )
        page.evaluate(f"window.scrollTo(0, {top} + ({h} - innerHeight) * 0.4)")
        page.wait_for_timeout(500)
        step = page.inner_text(".how-count b")
        check("how it works: scrolling moves the drawing", step not in ("00", "01"), f"step {step}")
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
        page.locator(".btn-make").click()
        wait_js(page, "/Picked/.test(document.querySelector('.results-status').textContent)", timeout=30_000)
        check("make faces, saved: picks from the bank", True, page.inner_text(".results-status"))

        # ------------------------------------------------------------ the AI check runs on this device, live or not
        page.locator("#check").scroll_into_view_if_needed()
        ex = page.locator(".example")
        ex.nth(0).click()
        wait_js(
            page,
            "['yes', 'no'].some(c => document.querySelector('.verdict').classList.contains(c))",
            timeout=240_000,
        )
        check(
            "AI check on this device: watermark found on a HumanDiT face",
            "yes" in (page.get_attribute(".verdict", "class") or ""),
            page.inner_text(".verdict-body")[:120],
        )
        ex.nth(1).click()
        wait_js(page, "document.querySelector('.verdict').classList.contains('no')", timeout=120_000)
        check("AI check on this device: none on the first version's face", True)

        # ------------------------------------------------------------ switch live mode on, through the page
        page.evaluate("window.scrollTo(0, 0)")
        page.locator(".gpu-strip").click()
        page.wait_for_timeout(800)
        name = page.inner_text(".live-device-name")
        check(
            "live dialog: the graphics chip by its model name",
            bool(re.search(r"NVIDIA|GeForce|Radeon|Intel .*Graphics|Apple|processor", name))
            and not re.search(r"\bgen[ -]?\d|turing", name, re.I),
            name,
        )
        check("live dialog: the measured devices", page.locator("dialog.live .lrow").count() >= 3)
        where = page.inner_text(".live-store")
        check(
            "live dialog: where the AI will be stored, and how to remove it",
            "CacheStorage" in where and "not your Downloads folder" in where,
            page.inner_text(".live-path"),
        )
        page.locator(".live-go").click()
        page.wait_for_timeout(700)
        check(
            "live dialog: please wait, step by step",
            page.is_visible(".live-wait") and "Please wait" in page.inner_text(".live-wait-title"),
            " | ".join(page.locator(".live-steps li").all_inner_texts()),
        )
        t0 = time.monotonic()
        # close it while the AI starts and press a button: it must say to wait, then run live
        page.locator(".live-close").click()
        check(
            "the pill says to wait while it starts",
            "Please wait" in page.inner_text(".live-pill .engine-text"),
            page.inner_text(".live-pill .engine-text"),
        )
        page.locator("#studio").scroll_into_view_if_needed()
        page.locator(".chip", has_text="Glasses").click()
        page.locator('.seg[data-name="count"] button[data-value="4"]').click()
        page.locator(".btn-make").click()
        page.wait_for_timeout(600)
        said = page.inner_text(".results-status")
        check(
            "a button pressed while it starts: please wait",
            "Please wait until the AI is running" in said,
            said[:90],
        )
        wait_js(page, "document.querySelector('.results-tag').classList.contains('live')", timeout=600_000)
        page.wait_for_timeout(1500)
        caps = page.locator(".face-cap").all_inner_texts()
        check(
            "make faces live: 4 faces with glasses, each confirmed",
            len(caps) == 4 and all("glasses" in c for c in caps) and page.locator(".mark.bad").count() == 0,
            f"{page.inner_text('.results-status')} | {caps[0] if caps else ''}",
        )
        check("make faces live: each face downloadable", page.locator(".face-dl").count() == 4)
        wait_js(
            page,
            "document.querySelector('.live-go span').textContent === 'Start making faces'",
            timeout=600_000,
        )
        check(
            "live mode: downloaded, prepared and timed",
            page.evaluate("document.documentElement.classList.contains('is-live')"),
            f"{time.monotonic() - t0:.0f} s; {page.inner_text('.live-size')}",
        )
        check("this device joins the ladder", page.locator(".ladder .lrow.you").count() == 1)

        # ------------------------------------------------------------ live: a feature Instant can't draw -> Best quality
        page.locator(".chip", has_text="Glasses").click()
        page.locator(".chip", has_text="Glasses").click()  # back to "any"
        page.locator(".chip", has_text="Goatee").click()
        page.wait_for_timeout(400)
        note = page.inner_text(".weak-note") if page.is_visible(".weak-note") else ""
        check("asking Instant for a goatee: says it uses Best quality", "Best quality" in note, note[:110])
        page.locator('.seg[data-name="count"] button[data-value="1"]').click()
        page.locator(".btn-make").click()
        wait_js(
            page,
            "/^Made .*Best quality/.test(document.querySelector('.results-status').textContent)",
            timeout=600_000,
        )
        cap = page.inner_text(".face-cap")
        check(
            "make faces live: a goatee, in Best quality, confirmed",
            "goatee" in cap and page.locator(".mark.bad").count() == 0,
            page.inner_text(".results-status")[:110],
        )
        page.locator(".chip", has_text="Goatee").click()
        page.locator(".chip", has_text="Goatee").click()  # back to "any"

        # ------------------------------------------------------------ live: change one thing
        page.locator("#edit").scroll_into_view_if_needed()
        page.locator(".btn-edit").click()
        wait_js(
            page, "document.querySelector('.edit-status').textContent.startsWith('Live')", timeout=180_000
        )
        check("change one thing: another person, live", True, page.inner_text(".edit-status"))

        # ------------------------------------------------------------ live: race on this GPU
        page.locator("#speed").scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        page.locator(".btn-race").click()
        wait_js(page, "document.querySelector('.race-hw').textContent.startsWith('Live')", timeout=600_000)
        with contextlib.suppress(TimeoutError):
            wait_js(page, "document.querySelectorAll('.lane.done').length === 4", timeout=120_000)
        check(
            "instant or best: 4 lanes raced live",
            page.locator(".lane.done").count() == 4,
            page.inner_text(".race-hw"),
        )

        page.locator("#numbers").scroll_into_view_if_needed()
        page.wait_for_timeout(1500)
        check("the numbers: 8 measured models", page.locator(".srow").count() == 8)
        page.evaluate("document.querySelector('.footer-town').scrollIntoView()")
        page.wait_for_timeout(1500)
        check("footer: the town draws", page.evaluate(LIT, ".footer-town canvas"))
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

        # ------------------------------------------------------------ remove the AI from this computer
        page.locator(".footer-meta [data-store]").click()
        page.wait_for_timeout(800)
        page.locator(".live-remove").click()
        check(
            "remove: asks once more",
            "Press again" in page.inner_text(".live-remove"),
            page.inner_text(".live-remove"),
        )
        page.locator(".live-remove").click()
        wait_js(page, "document.querySelector('.live-pill').dataset.state === 'saved'", timeout=30_000)
        page.wait_for_timeout(500)
        check(
            "remove: the browser's copy is gone, back to saved results",
            not page.evaluate("caches.has('humandit-models')"),
            page.inner_text(".live-store-what")[:100],
        )
        page.locator(".live-close").click()

        # ------------------------------------------------------------ phone
        phone = b.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        phone.on("pageerror", lambda e: errors.append(str(e)))
        phone.goto(a.url, wait_until="load")
        phone.wait_for_timeout(3000)
        over = phone.evaluate("document.documentElement.scrollWidth - innerWidth")
        check("phone: no sideways scrolling", over <= 0, f"{over}px")
        phone.close()

        real = [e for e in errors if "favicon" not in e and "Failed to load resource" not in e]
        check("no JavaScript errors", not real, " | ".join(real)[:300])
        gone = [u for u in missing if "favicon" not in u]
        check("no missing files", not gone, " ".join(gone)[:300])
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
