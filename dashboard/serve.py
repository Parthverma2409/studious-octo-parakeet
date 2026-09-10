"""Optional local server for the dashboard.

The dashboard is designed to work from `file://` and needs no server at all. This
exists only for people who prefer a URL (and for browsers configured to distrust
local files). Stdlib only, no dependencies, no reload.

Usage:
    python dashboard/serve.py            # serves dashboard/ on port 8000
    python dashboard/serve.py --port 9001
"""

from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dashboard/serve.py")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--root", type=Path, default=DASHBOARD_DIR)
    args = parser.parse_args(argv)

    if not (args.root / "data" / "index.js").exists():
        print("warning: %s missing -- run: python dashboard/build_index.py" % (args.root / "data" / "index.js"))

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(args.root))
    with socketserver.TCPServer((args.bind, args.port), handler) as httpd:
        print("serving %s at %s:%d  (ctrl-c to stop)" % (args.root, args.bind, args.port))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
