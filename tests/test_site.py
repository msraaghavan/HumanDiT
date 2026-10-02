"""The static website (site/): every file the page loads exists, the saved results are complete, the in-browser AI's
pieces are in place, and the Cloudflare build writes a strict content security policy with cross-origin isolation."""

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
    worker = (SITE / "js" / "ai" / "worker.js").read_text(encoding="utf-8")
    vendored = re.search(r'from "\.\./\.\./(vendor/[^"]+)"', worker).group(1)
    assert (SITE / vendored).exists()
    assert (SITE / "vendor" / "ort" / "ort-wasm-simd-threaded.asyncify.wasm.gz").exists()
    for f in ("inter", "jetbrains-mono"):  # served with the page: no third-party requests
        assert (SITE / "fonts" / f"{f}.woff2").exists()
    assert "fonts.googleapis.com" not in html


def test_gate_runs_before_anything_draws():
    """Phones, tablets and computers without a usable graphics chip get js/gate.js's screen: a classic script in
    <head>, before the stylesheet and the page's module, in older JavaScript an old phone can still run."""
    html = (SITE / "index.html").read_text(encoding="utf-8")
    head = html.split("</head>")[0]
    assert (
        head.index('<script src="js/gate.js"></script>')
        < head.index("css/site.css")
        < head.index("js/main.js")
    )
    gate = (SITE / "js" / "gate.js").read_text(encoding="utf-8")
    code = re.sub(r"//[^\n]*|\"[^\"\n]*\"|'[^'\n]*'", "", gate)  # comments and strings aside
    assert not re.search(r"=>|\?\.|\?\?|\bconst\b|\blet\b|`|\basync\b|\bclass\b", code)
    assert (SITE / "assets" / "gate.jpg").exists()
    main = (SITE / "js" / "main.js").read_text(encoding="utf-8")
    assert "HDGate" in main and "gate.ready" in main  # the website waits for the gate's answer


def test_saved_results_are_complete():
    site = json.loads((SITE / "assets" / "site.json").read_text())
    assert len(site["features"]) == 12
    assert len(site["scoreboard"]) == 8
    pre = site["preset"]
    files = [h["src"] for h in pre["hero"]] + [f["src"] for f in pre["studio"]["faces"]]
    files += [r["img"] for r in pre["compare"]["runs"]]
    bank = json.loads((SITE / "assets" / "bank.json").read_text())
    assert not {"blend", "edits"} & set(pre) and not {"blends", "edits"} & set(
        bank
    )  # both removed from the site
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
            "--github",
            "https://github.com/someone/HumanDiT",
            "--no-models",
            "--out",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    cfg = (out / "config.js").read_text(encoding="utf-8")
    assert "https://github.com/someone/HumanDiT" in cfg
    assert json.loads((out / "wrangler.jsonc").read_text())["name"] == "humandit"
    assert "wrangler.jsonc" in (out / ".assetsignore").read_text()
    assert not (out / "models").exists()
    headers = (out / "_headers").read_text(encoding="utf-8")
    assert "script-src 'self' 'wasm-unsafe-eval';" in headers and "'unsafe-eval'" not in headers
    assert "connect-src 'self' blob:;" in headers and "huggingface" not in headers
    assert "Cross-Origin-Embedder-Policy: require-corp" in headers  # threads for the processor fallback


def test_model_manifest_if_exported():
    """site/models is built by scripts/site/export_web_models.py (not committed); when present it must be whole."""
    man = SITE / "models" / "manifest.json"
    if not man.exists():
        return
    m = json.loads(man.read_text())
    for key, f in m["files"].items():
        parts = [SITE / "models" / m["version"] / f"{key}.{i:02d}" for i in range(f["parts"])]
        assert all(q.exists() for q in parts), key
        assert sum(q.stat().st_size for q in parts) == f["bytes"], key
        assert all(q.stat().st_size <= 25 * 2**20 for q in parts), key  # Cloudflare's file limit
    assert set(m["packs"]["core"]) == {"generator", "decoder", "tm_encoder"}
    assert len(m["watermark"]["bits"]) == 100
    assert len(m["features"]) == 12
