"""The static website (site/): every file the page loads exists, the saved results are complete, and the Cloudflare
build fills in the engine and a strict content security policy."""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def test_page_references_exist():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    refs = re.findall(r'(?:src|href|data-src)="([^"#:]+)"', html)
    assert refs, "no local references found"
    missing = [r for r in refs if not (SITE / r).exists()]
    assert not missing, missing
    js = (SITE / "js" / "engine.js").read_text(encoding="utf-8")
    vendored = re.search(r'from "\.\./(vendor/[^"]+)"', js).group(1)
    assert (SITE / vendored).exists()


def test_saved_results_are_complete():
    site = json.loads((SITE / "assets" / "site.json").read_text())
    assert len(site["features"]) == 12
    assert len(site["scoreboard"]) == 8
    pre = site["preset"]
    files = [h["src"] for h in pre["hero"]] + [f["src"] for f in pre["studio"]["faces"]]
    files += [e[k] for e in pre["edits"] for k in ("without", "choice", "with")]
    files += [pre["blend"]["src"]] + [r["img"] for r in pre["compare"]["runs"]]
    assert all((SITE / f).exists() for f in files), [f for f in files if not (SITE / f).exists()]
    for h in pre["hero"]:  # a static amount per step, from all static down towards none
        assert len(h["noise_left"]) == h["n"] == 25
        assert h["noise_left"][0] == 1.0 and h["noise_left"] == sorted(h["noise_left"], reverse=True)
    assert [r["steps"] for r in pre["compare"]["runs"]] == [1, 10, 25, 50]


def test_cloudflare_build(tmp_path):
    out = tmp_path / "site"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/site/build_site.py"),
            "--space",
            "someone/HumanDiT-engine",
            "--model",
            "someone/HumanDiT",
            "--out",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    cfg = (out / "config.js").read_text(encoding="utf-8")
    assert 'space: "someone/HumanDiT-engine"' in cfg and "__SPACE__" not in cfg
    assert "https://huggingface.co/someone/HumanDiT" in cfg
    headers = (out / "_headers").read_text(encoding="utf-8")
    assert "script-src 'self';" in headers and "unsafe-eval" not in headers
    assert "connect-src 'self' https://huggingface.co https://*.hf.space" in headers
