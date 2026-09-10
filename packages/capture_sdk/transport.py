"""The seam where capture is allowed to fail.

Everything that can go wrong between the agent and the ingest service -- DNS, a
refused connection, a hung socket, a 500 -- goes wrong behind this protocol and
on a background thread. Two consequences the rest of the SDK depends on:

  * UC4 E1 (ingest unreachable) is reproducible in a test by handing the client
    a transport that refuses, with no service to stop and no port to guess.
  * A transport failure has exactly one blast radius, the worker thread, so it
    can never reach the user-facing call path.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

#: A 4xx means this trajectory will never be accepted, however often it is sent.
#: These two are the exceptions: the request was fine, the server was not ready.
_RETRYABLE_STATUS = frozenset({408, 429})


class TransportError(RuntimeError):
    """Delivery failed.

    `permanent` says retrying is pointless -- the ingest service rejected the
    payload itself and has already quarantined it, so a retry would only add a
    duplicate quarantine entry for the same defect.
    """

    def __init__(self, message: str, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent


@runtime_checkable
class Transport(Protocol):
    def send(self, trajectory: dict) -> None: ...


class HttpTransport:
    """POSTs one trajectory as JSON. Blocking on purpose -- it runs on the worker."""

    def __init__(self, url: str, timeout: float = 2.0):
        self.url = url
        self.timeout = timeout

    def send(self, trajectory: dict) -> None:
        body = json.dumps(trajectory).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status >= 300:
                    raise TransportError("ingest returned %d" % response.status)
        except urllib.error.HTTPError as exc:
            permanent = 400 <= exc.code < 500 and exc.code not in _RETRYABLE_STATUS
            raise TransportError("ingest returned %d" % exc.code, permanent=permanent) from exc
        except urllib.error.URLError as exc:
            # Connection refused, DNS failure, timeout: the service may come back.
            raise TransportError("ingest unreachable: %s" % exc.reason) from exc
        except OSError as exc:
            raise TransportError("ingest unreachable: %s" % exc) from exc


class MemoryTransport:
    """Collects delivered trajectories in-process. For demos and tests."""

    def __init__(self) -> None:
        self.received: list[dict] = []
        self._lock = threading.Lock()

    def send(self, trajectory: dict) -> None:
        with self._lock:
            self.received.append(trajectory)

    def __len__(self) -> int:
        with self._lock:
            return len(self.received)
