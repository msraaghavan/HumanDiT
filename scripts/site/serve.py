"""Serve site/ locally the way Cloudflare Pages will: with the same Content-Security-Policy and headers.

    uv run python scripts/site/serve.py [--port 8081] [--engine http://localhost:7861]

The policy is the one scripts/site/build_site.py writes to _headers, with the local engine allowed in place of
Hugging Face, so scripts/site/site_check.py can prove the page works under it before it is deployed.
"""

from __future__ import annotations

import argparse
import functools
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_site import CSP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--engine", default="http://localhost:7861")
    ap.add_argument("--dir", default="site", help="the folder to serve: site/, or a build such as dist/site")
    a = ap.parse_args()
    ws = a.engine.replace("http", "ws", 1)
    csp = CSP.replace("connect-src 'self'", f"connect-src 'self' {a.engine} {ws}").replace(
        "img-src 'self'", f"img-src 'self' {a.engine}"
    )

    class Handler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Content-Security-Policy", csp)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *_):
            pass

    site = HERE.parents[1] / a.dir
    server = ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(Handler, directory=str(site)))
    print(f"serving {site} with the production CSP on http://localhost:{a.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
