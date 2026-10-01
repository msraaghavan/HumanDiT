"""Assemble the website for Cloudflare: dist/site, with the AI's models and the links filled in.

    uv run --extra demo python scripts/site/export_web_models.py   # once: site/models/ (not committed)
    uv run python scripts/site/build_site.py [--github https://github.com/USER/REPO]

Then deploy the folder, after logging in to Cloudflare yourself (`npx wrangler login`):

    npx wrangler deploy --cwd dist/site

The folder carries its own wrangler.jsonc (a Worker named humandit that serves these static files; Cloudflare Pages
is now part of Workers) and an .assetsignore, so that config is never published as a page. There is no server: the
models run in each visitor's browser, on their graphics chip or processor.

Also writes Cloudflare's _headers: a content security policy that lets the page load nothing from anywhere else,
cross-origin isolation (the AI's multi-threaded processor fallback needs it), and long caching for the versioned
model parts. Nothing is uploaded by this script.
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
        "script-src 'self' 'wasm-unsafe-eval'",  # ONNX Runtime Web compiles its WebAssembly
        "worker-src 'self'",
        "style-src 'self'",
        "style-src-attr 'unsafe-inline'",  # the HTML's per-element animation delays (style="--d:.5s")
        "font-src 'self'",
        "img-src 'self' data: blob:",  # faces made on the device are blobs
        "media-src 'self'",
        "connect-src 'self' blob:",  # reading back a face made on this device (blob:)
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
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
/vendor/*
  Cache-Control: public, max-age=31536000, immutable
/models/*/*
  Cache-Control: public, max-age=31536000, immutable
/fonts/*
  Cache-Control: public, max-age=2592000
/assets/*
  Cache-Control: public, max-age=86400
"""

ISOLATION = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", help="https://github.com/USER/REPO, for the link to the code")
    ap.add_argument("--model-card", help="a URL for the model card, if it is published somewhere")
    ap.add_argument("--worker", default="humandit", help="the Cloudflare Worker's name (its URL)")
    ap.add_argument("--out", default="dist/site")
    ap.add_argument("--no-models", action="store_true", help="build without site/models (saved results only)")
    a = ap.parse_args()
    out = ROOT / a.out
    if not (a.no_models or (SITE / "models" / "manifest.json").exists()):
        raise SystemExit(
            "site/models is missing: run scripts/site/export_web_models.py (or pass --no-models)"
        )
    shutil.rmtree(out, ignore_errors=True)
    skip = ["*.map", ".DS_Store", ".gitignore"] + (["models"] if a.no_models else [])
    shutil.copytree(SITE, out, ignore=shutil.ignore_patterns(*skip))
    links = {"github": a.github or "", "model": a.model_card or ""}
    cfg = (SITE / "config.js").read_text(encoding="utf-8")
    cfg = re.sub(r"links: \{[^}]*\}", "links: " + json.dumps(links), cfg)
    (out / "config.js").write_text(cfg, encoding="utf-8")
    (out / "_headers").write_text(HEADERS, encoding="utf-8")
    worker = {"name": a.worker, "compatibility_date": "2026-09-26", "assets": {"directory": "."}}
    (out / "wrangler.jsonc").write_text(json.dumps(worker, indent=2) + "\n", encoding="utf-8")
    (out / ".assetsignore").write_text("wrangler.jsonc\n.wrangler\n.assetsignore\n", encoding="utf-8")
    missing = [f for f in ("assets/site.json", "assets/og.jpg", "assets/faces.jpg") if not (out / f).exists()]
    if missing:
        raise SystemExit(f"missing {missing}: run scripts/site/build_assets.py and og_image.py first")
    files = [f for f in out.rglob("*") if f.is_file()]
    big = [str(f.relative_to(out)) for f in files if f.stat().st_size > 25 * 2**20]
    if big:
        raise SystemExit(f"Cloudflare serves files up to 25 MiB; too big: {big}")
    print(
        f"{out}: {len(files)} files, {sum(f.stat().st_size for f in files) / 2**20:.1f} MB; "
        f"models {'none (saved results only)' if a.no_models else 'included'}"
    )


if __name__ == "__main__":
    main()
