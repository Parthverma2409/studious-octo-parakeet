"""The ingest endpoint, on the standard library.

FastAPI is what the SRS names and what Phase 5 will use; `http.server` is what
is installed here, and the interesting behaviour -- accept, quarantine, expose
the quarantine -- is in `pipeline.py` where no web framework can reach it. This
module is the thin part, and swapping it is a rewrite of about sixty lines.

It runs threaded and on a background thread so a test can start it, point a real
`CaptureClient` at it over a real socket, and stop it again. Binding port 0 by
default gives every test its own port instead of a shared constant that fails
when two runs overlap.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from packages.eval_ingest.pipeline import IngestPipeline

TRAJECTORY_PATH = "/v1/trajectories"
QUARANTINE_PATH = "/v1/quarantine"
STATS_PATH = "/v1/stats"
HEALTH_PATH = "/healthz"

#: Refuse absurd bodies before decoding them. A trajectory is a few kilobytes;
#: anything near this is a bug or an attack, and either way must not be read
#: into the memory of the process the assistant depends on.
MAX_BODY_BYTES = 1 << 20


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _drain(self) -> None:
        """Read and discard the request body before an early error response.

        On a keep-alive connection the client may still be writing the body when
        we decide to reject the request. Answering without consuming it leaves
        unread bytes in the socket, and closing over them makes Windows reset the
        connection -- the client then sees ConnectionAbortedError instead of the
        status code we actually sent. This surfaced as a test that passed alone
        and failed under load, which is the worst way for it to surface.
        """
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        remaining = min(length, MAX_BODY_BYTES)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)

    def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
        if self.path.rstrip("/") != TRAJECTORY_PATH:
            self._drain()
            self._send(404, {"error": "no such endpoint: %s" % self.path})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._drain()
            self._send(400, {"error": "bad Content-Length"})
            return
        if length > MAX_BODY_BYTES:
            self._drain()
            self._send(413, {"error": "body too large"})
            return

        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            # Not quarantinable: there is no trajectory here to hold, only bytes.
            self._send(400, {"error": "malformed JSON: %s" % exc})
            return

        result = self.server.pipeline.submit(payload, source=self.client_address[0])
        # 422, not 500: the request was understood and the payload was kept. The
        # capture SDK reads 4xx as permanent and stops retrying, which is right
        # -- a schema-invalid trajectory does not become valid on the third try.
        self._send(202 if result.accepted else 422, result.as_dict())

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler's naming
        path = self.path.rstrip("/") or "/"
        pipeline = self.server.pipeline
        if path == HEALTH_PATH:
            self._send(200, {"ok": True})
        elif path == QUARANTINE_PATH:
            records = pipeline.quarantine.all()
            self._send(
                200,
                {
                    "count": len(records),
                    "total_ever": pipeline.quarantine.total_ever,
                    "items": [r.as_dict() for r in records],
                },
            )
        elif path == STATS_PATH:
            self._send(200, pipeline.snapshot())
        else:
            self._send(404, {"error": "no such endpoint: %s" % self.path})

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence. The default writes a line to stderr per request, which turns
        a throughput test into a wall of noise."""

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class IngestService:
    """Starts and stops the endpoint. Usable as a context manager."""

    def __init__(self, pipeline: IngestPipeline | None = None, host: str = "127.0.0.1", port: int = 0):
        self.pipeline = pipeline or IngestPipeline()
        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._server.pipeline = self.pipeline
        self._thread = threading.Thread(target=self._server.serve_forever, name="eval-ingest", daemon=True)
        self._started = False

    def start(self) -> "IngestService":
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def stop(self) -> None:
        if self._started:
            self._server.shutdown()
            self._thread.join(timeout=5.0)
            self._started = False
        self._server.server_close()

    @property
    def host(self) -> str:
        return self._server.server_address[0]

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def url(self) -> str:
        """The endpoint a `CaptureClient`'s HttpTransport should be given."""
        return "http://%s:%d%s" % (self.host, self.port, TRAJECTORY_PATH)

    def __enter__(self) -> "IngestService":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
