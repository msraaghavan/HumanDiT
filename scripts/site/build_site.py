"""Assemble the website for Cloudflare Pages: dist/site, with the engine's Space and the links filled in.

    uv run python scripts/site/build_site.py --space USER/HumanDiT-engine --model USER/HumanDiT \
        [--github https://github.com/USER/REPO]

Then deploy the folder, after logging in to Cloudflare yourself (`npx wrangler login`):

    npx wrangler pages deploy dist/site --project-name humandit

Also writes Cloudflare's _headers: a content security policy that lets the page talk only to Hugging Face (the
engine) and Google Fonts, and long caching for the vendored client. Nothing is uploaded by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1].parent
SITE = ROOT / "site"

CSP = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' https://fonts.googleapis.com",
        "style-src-attr 'unsafe-inline'",  # the HTML's per-element animation delays (style="--d:.5s")
        "font-src https://fonts.gstatic.com",
        "img-src 'self' data: blob: https://*.hf.space",
        "media-src 'self'",
        "connect-src 'self' https://huggingface.co https://*.hf.space wss://*.hf.space",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'none'",
        "object-src 'none'",
    ]
)

HEADERS = f"""/*
  Content-Security-Policy: {CSP}
  X-Content-Type-Options: nosniff
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=(), interest-cohort=()
/vendor/*
  Cache-Control: public, max-age=31536000, immutable
/assets/*
  Cache-Control: public, max-age=86400
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="USER/NAME of the Hugging Face Space running app/api.py")
    ap.add_argument("--model", help="USER/NAME of the model repo, for the link to the model card")
    ap.add_argument("--github", help="https://github.com/USER/REPO, for the link to the code")
    ap.add_argument("--out", default="dist/site")
    a = ap.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", a.space):
        raise SystemExit(f"--space should look like USER/NAME, not {a.space!r}")
    out = ROOT / a.out
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns("*.map", ".DS_Store"))
    links = {
        "github": a.github or "",
        "model": f"https://huggingface.co/{a.model}" if a.model else "",
        "space": f"https://huggingface.co/spaces/{a.space}",
    }
    cfg = (SITE / "config.js").read_text(encoding="utf-8")
    cfg = cfg.replace('space: "__SPACE__"', f"space: {json.dumps(a.space)}")
    cfg = re.sub(r"links: \{[^}]*\}", "links: " + json.dumps(links).replace('":', '": '), cfg)
    assert "__SPACE__" not in cfg
    (out / "config.js").write_text(cfg, encoding="utf-8")
    (out / "_headers").write_text(HEADERS, encoding="utf-8")
    missing = [f for f in ("assets/site.json", "assets/og.jpg", "assets/faces.jpg") if not (out / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run scripts/site/build_assets.py and og_image.py first")
    files = [f for f in out.rglob("*") if f.is_file()]
    print(
        f"{out}: {len(files)} files, {sum(f.stat().st_size for f in files) / 2**20:.1f} MB; engine {a.space}"
    )


if __name__ == "__main__":
    main()
