"""Which graphics chip the website's AI runs on, browser by browser, on this computer.

    uv run python scripts/site/serve.py --port 8081 &
    uv run python scripts/site/gpu_choice.py [--url http://localhost:8081/] [--channels msedge chrome]

For each browser, twice: as installed (on a Windows laptop with two chips, Chrome and Edge run on the built-in one)
and with the "Force High Performance GPU" setting the page tells people about. Each run starts live mode through the
page's own button in a fresh profile and reports the chip the page names, whether it shows the advice to switch, the
chip the AI says it runs on, and the measured seconds a face from the page's own speed test.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time

from playwright.sync_api import sync_playwright


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    ap.add_argument("--channels", nargs="+", default=["msedge", "chrome"])
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with sync_playwright() as p:
        for channel in a.channels:
            for forced in (False, True):
                args = ["--enable-unsafe-webgpu"] + (["--force_high_performance_gpu"] if forced else [])
                with tempfile.TemporaryDirectory() as profile:
                    ctx = p.chromium.launch_persistent_context(
                        profile, channel=channel, args=args, viewport={"width": 1440, "height": 900}
                    )
                    pg = ctx.pages[0] if ctx.pages else ctx.new_page()
                    errors: list[str] = []
                    pg.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                    pg.goto(a.url, wait_until="load")
                    pg.wait_for_timeout(3500)  # the hero line's device check runs after a short idle
                    strip = pg.inner_text(".gpu-strip-text")
                    pg.locator(".gpu-strip").click()
                    pg.wait_for_timeout(1500)
                    named = pg.inner_text(".live-device-name")
                    tip = pg.locator(".live-tip").is_visible()
                    pg.locator(".live-go").click()
                    t0 = time.monotonic()
                    while "Start making faces" not in pg.inner_text(".live-go"):
                        if time.monotonic() - t0 > 900 or "Try again" in pg.inner_text(".live-go"):
                            break
                        pg.wait_for_timeout(1000)
                    took = time.monotonic() - t0
                    print(
                        f"{channel:7s} {'forced to the fast chip' if forced else 'as installed':24s}| page names: {named}\n"
                        f"{'':32s}| switch advice: {'shown' if tip else 'not shown'}; opening line: {strip[:110]}\n"
                        f"{'':32s}| AI: {pg.inner_text('.live-size')}  (start-up {took:.0f} s)"
                        + (f"\n{'':32s}| errors: {errors}" if errors else ""),
                        flush=True,
                    )
                    ctx.close()


if __name__ == "__main__":
    main()
