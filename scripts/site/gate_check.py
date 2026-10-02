"""Who gets the website and who gets the "needs a laptop" screen (site/js/gate.js), in a real browser.

    uv run python scripts/site/serve.py --port 8081 &
    uv run python scripts/site/gate_check.py [--url http://localhost:8081/] [--shots DIR]

Phones and tablets as they really report themselves (Playwright's device profiles), the same devices asking for the
"desktop site" (an iPad that says it's a Mac, an Android phone that says it's a Linux PC), a TV, Windows laptops with
touch screens (which must pass), a Mac, and computers whose browser has no WebGPU, no graphics chip, or a software
one. For each: gated or not, which screen, no JavaScript errors, nothing wider than the screen, and on a gated device
that the website itself never started (no AI files, no hero pictures). --shots saves a screenshot of each screen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

MAC_SAFARI = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15"
MAC_CHROME = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
LINUX_CHROME = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
WIN_EDGE = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0"
WIN_FIREFOX = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0"
TIZEN_TV = "Mozilla/5.0 (SMART-TV; LINUX; Tizen 7.0) AppleWebKit/537.36 (KHTML, like Gecko) 108.0.5359.1/7.0 TV Safari/537.36"


def js(*lines: str) -> str:
    return "\n".join(lines)


def touch_points(n: int) -> str:
    return f"Object.defineProperty(Navigator.prototype, 'maxTouchPoints', {{ get: () => {n} }});"


def platform(s: str) -> str:
    return f"Object.defineProperty(Navigator.prototype, 'platform', {{ get: () => {s!r} }});"


def no_uad() -> str:
    return "Object.defineProperty(Navigator.prototype, 'userAgentData', { get: () => undefined });"


def screen(w: int, h: int) -> str:
    return (
        f"Object.defineProperty(Screen.prototype, 'width', {{ get: () => {w} }});"
        f"Object.defineProperty(Screen.prototype, 'height', {{ get: () => {h} }});"
    )


def adapter(info: dict | None, fallback: bool = False) -> str:
    """Replace what WebGPU answers: no chip (None), or a chip that reports `info`."""
    if info is None:
        return "GPU.prototype.requestAdapter = async () => null;"
    import json

    return (
        "GPU.prototype.requestAdapter = async () => ({ info: Object.assign("
        + json.dumps(info)
        + f", {{ isFallbackAdapter: {str(fallback).lower()} }}), features: new Set(), limits: {{}}, requestDevice: async () => {{ throw new Error('test') }} }});"
    )


NO_WEBGPU = "delete Navigator.prototype.gpu;"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8081/")
    ap.add_argument("--channel", default="msedge")
    ap.add_argument("--shots", help="a folder for a screenshot of each gate screen")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    shots = Path(a.shots) if a.shots else None
    if shots:
        shots.mkdir(parents=True, exist_ok=True)
    fails = 0

    with sync_playwright() as p:
        dev = p.devices

        def profile(name: str) -> dict:
            d = dict(dev[name])
            d.pop("default_browser_type", None)
            return d

        # name, context options, init script, expected kind ("" = the website), query
        cases: list[tuple[str, dict, str, str, str]] = [
            (
                "this laptop (Edge, its own graphics chip)",
                {"viewport": {"width": 1440, "height": 900}},
                "",
                "",
                "",
            ),
            ("iPhone 15", profile("iPhone 15"), "", "phone", ""),
            ("Pixel 7", profile("Pixel 7"), "", "phone", ""),
            (
                "Galaxy S24",
                profile("Galaxy S24") if "Galaxy S24" in dev else profile("Galaxy S9+"),
                "",
                "phone",
                "",
            ),
            ("iPad Pro 11", profile("iPad Pro 11"), "", "tablet", ""),
            ("Galaxy Tab S4", profile("Galaxy Tab S4"), "", "tablet", ""),
            (
                "iPad asking for the desktop site (says it's a Mac)",
                {"user_agent": MAC_SAFARI, "viewport": {"width": 1180, "height": 820}, "has_touch": True},
                touch_points(5) + no_uad() + screen(820, 1180),
                "tablet",
                "",
            ),
            (
                "iPhone asking for the desktop site (says it's a Mac)",
                {
                    "user_agent": MAC_SAFARI,
                    "viewport": {"width": 980, "height": 1800},
                    "has_touch": True,
                    "is_mobile": True,
                },
                touch_points(5) + no_uad() + screen(393, 852),
                "phone",
                "",
            ),
            (
                "Android phone, desktop site (Linux PC user agent, ARM processor)",
                {
                    "user_agent": LINUX_CHROME,
                    "viewport": {"width": 980, "height": 1900},
                    "has_touch": True,
                    "is_mobile": True,
                },
                touch_points(5) + platform("Linux aarch64") + no_uad() + screen(412, 915),
                "phone",
                "",
            ),
            (
                "Android phone, desktop site, every software clue hidden (touch-only pointer)",
                {
                    "user_agent": LINUX_CHROME,
                    "viewport": {"width": 980, "height": 1900},
                    "has_touch": True,
                    "is_mobile": True,
                },
                touch_points(5) + platform("Linux x86_64") + no_uad() + screen(412, 915),
                "phone",
                "",
            ),
            (
                "Android tablet, desktop site, keyboard and trackpad attached (only its Mali chip gives it away)",
                {"user_agent": LINUX_CHROME, "viewport": {"width": 1280, "height": 800}, "has_touch": True},
                touch_points(10)
                + platform("Linux x86_64")
                + no_uad()
                + screen(800, 1280)
                + adapter({"vendor": "arm", "architecture": "valhall"}),
                "tablet",
                "",
            ),
            (
                "smart TV",
                {"user_agent": TIZEN_TV, "viewport": {"width": 1920, "height": 1080}},
                no_uad(),
                "other",
                "",
            ),
            (
                "Windows laptop with a touch screen",
                {"user_agent": WIN_EDGE, "viewport": {"width": 1366, "height": 768}, "has_touch": True},
                touch_points(10),
                "",
                "",
            ),
            (
                "Windows on Snapdragon, touch screen (an Adreno chip, but a PC)",
                {"user_agent": WIN_EDGE, "viewport": {"width": 1366, "height": 768}, "has_touch": True},
                touch_points(10) + adapter({"vendor": "qualcomm", "architecture": "adreno-8xx"}),
                "",
                "",
            ),
            (
                "a Mac (Chrome)",
                {"user_agent": MAC_CHROME, "viewport": {"width": 1512, "height": 945}},
                no_uad(),
                "",
                "",
            ),
            (
                "a Mac with an older Safari (no WebGPU)",
                {"user_agent": MAC_SAFARI, "viewport": {"width": 1512, "height": 945}},
                no_uad() + NO_WEBGPU,
                "browser",
                "",
            ),
            (
                "Windows, Firefox without WebGPU",
                {"user_agent": WIN_FIREFOX, "viewport": {"width": 1366, "height": 768}},
                no_uad() + NO_WEBGPU,
                "browser",
                "",
            ),
            (
                "Windows, WebGPU but no graphics chip answers",
                {"viewport": {"width": 1366, "height": 768}},
                adapter(None),
                "nogpu",
                "",
            ),
            (
                "Windows, graphics acceleration off (a software chip)",
                {"viewport": {"width": 1366, "height": 768}},
                adapter({"vendor": "google", "architecture": "swiftshader"}, fallback=True),
                "software",
                "",
            ),
            (
                "Linux, Chrome, no graphics chip answers",
                {"user_agent": LINUX_CHROME, "viewport": {"width": 1366, "height": 768}},
                no_uad() + adapter(None),
                "nogpu",
                "",
            ),
            (
                "preview of the phone screen on this laptop (?gate=phone)",
                {"viewport": {"width": 1440, "height": 900}},
                "",
                "phone",
                "?gate=phone",
            ),
            (
                "iPhone 15, reduced motion",
                {**profile("iPhone 15"), "reduced_motion": "reduce"},
                "",
                "phone",
                "",
            ),
        ]

        b = p.chromium.launch(
            channel=a.channel, args=["--enable-unsafe-webgpu", "--force_high_performance_gpu"]
        )
        for name, opts, init, want, query in cases:
            ctx = b.new_context(**opts)
            if init:
                ctx.add_init_script(init)
            page = ctx.new_page()
            errors: list[str] = []
            asked: list[str] = []
            page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
            page.on("console", lambda m, errors=errors: errors.append(m.text) if m.type == "error" else None)
            page.on("request", lambda r, asked=asked: asked.append(r.url))
            page.goto(a.url + query, wait_until="load")
            page.wait_for_timeout(2500)
            got = page.evaluate(
                """() => ({ gated: !!(window.HDGate && HDGate.gated), kind: (window.HDGate && HDGate.kind) || '',
                  title: document.title, gate: !!document.querySelector('.gate'),
                  site: getComputedStyle(document.querySelector('body > main:not(.gate)')).display,
                  faces: document.querySelectorAll('.gate-face').length,
                  over: Math.max(innerWidth, document.documentElement.scrollWidth) - visualViewport.width,
                  where: (document.querySelector('.gate-where') || {}).textContent || '',
                  ask: (document.querySelector('.gate-ask') || {}).textContent || '' })"""
            )
            started = [u for u in asked if "/models/" in u or "/assets/hero/" in u or "hero.json" in u]
            problems = []
            if got["kind"] != want:
                problems.append(f"expected {want or 'the website'}, got {got['kind'] or 'the website'}")
            if want:
                if not got["gate"] or got["site"] != "none":
                    problems.append("the gate screen isn't the only thing shown")
                if got["faces"] != 9:
                    problems.append(f"{got['faces']} faces in the box")
                if started:
                    problems.append(f"the website started anyway: {started[:2]}")
            elif got["gate"]:
                problems.append("a gate screen on a computer that should pass")
            if got["over"] > 1:
                problems.append(f"{got['over']:.0f}px wider than the screen")
            real = [e for e in errors if "favicon" not in e]
            if real:
                problems.append("errors: " + " | ".join(real)[:200])
            ok = not problems
            fails += not ok
            detail = f"{got['where']} {got['ask'][:90]}" if want else ""
            print(
                f"{'PASS' if ok else 'FAIL'}  {name}: {want or 'the website'}  {detail}{'  <- ' + '; '.join(problems) if problems else ''}",
                flush=True,
            )
            if shots and want:
                page.screenshot(
                    path=str(shots / f"{name.split(' (')[0].replace(' ', '_').replace(',', '')[:40]}.png")
                )
            ctx.close()
        b.close()
    print(f"\n{len(cases) - fails} of {len(cases)} passed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
