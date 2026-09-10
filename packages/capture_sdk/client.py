"""FR7 capture that cannot slow the assistant down (UC4 E1).

The headline property: when the ingest service is unreachable, the Sync Agent's
user-facing response is neither blocked nor slowed. That is not a matter of
being fast -- it is a matter of the agent's thread never touching the network at
all. `capture` appends to an in-memory buffer under a lock it holds for a few
microseconds and returns; a single background worker owns every send, every
retry and every backoff.

The corollary is that capture is best-effort by construction. An evaluation
platform that can stall a booking assistant is a worse trade than a gap in the
evaluation record, so the buffer is bounded, the losses are counted, and
`metrics()` reports them rather than hiding them.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from packages.capture_sdk.buffer import BoundedBuffer
from packages.capture_sdk.transport import Transport, TransportError


@dataclass(frozen=True)
class CaptureMetrics:
    """A snapshot. Every loss the SDK can cause has a counter here."""

    captured: int = 0
    sent: int = 0
    dropped: int = 0
    rejected: int = 0
    retries: int = 0
    abandoned: int = 0
    refused: int = 0
    pending: int = 0
    last_error: str | None = None

    @property
    def lost(self) -> int:
        """Trajectories the platform will never see, whatever the reason."""
        return self.dropped + self.rejected + self.abandoned + self.refused


class CaptureClient:
    """Buffers trajectories locally and forwards them on a background thread."""

    def __init__(
        self,
        transport: Transport,
        capacity: int = 512,
        max_attempts: int = 3,
        retry_base_delay: float = 0.05,
        retry_max_delay: float = 1.0,
        poll_interval: float = 0.05,
        name: str = "capture-sdk",
    ):
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._transport = transport
        self._buffer = BoundedBuffer(capacity)
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._counts = {"captured": 0, "sent": 0, "rejected": 0, "retries": 0, "abandoned": 0, "refused": 0}
        self._inflight = 0
        self._last_error: str | None = None
        self._stopping = threading.Event()
        # Daemon: a forgotten close() must not keep the assistant's process alive.
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- caller side (must stay off the network) ---------------------------

    def capture(self, trajectory: dict) -> bool:
        """Hands a trajectory to the worker. Never blocks, never raises.

        Returns False when the trajectory was refused or displaced something
        older -- information for the metrics, never an error for the caller to
        handle, because the caller is in the middle of answering a user.
        """
        if self._stopping.is_set():
            self._bump("refused")
            return False
        self._bump("captured")
        return self._buffer.put(trajectory)

    def flush(self, timeout: float = 2.0) -> bool:
        """Waits for the buffer to empty. False if it did not within `timeout`.

        Polled rather than signalled: flush is a test and shutdown affordance,
        not a hot path, and polling keeps the capture path free of a second
        condition variable to get wrong.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                idle = self._inflight == 0
            if idle and len(self._buffer) == 0:
                return True
            time.sleep(0.002)
        with self._lock:
            return self._inflight == 0 and len(self._buffer) == 0

    def close(self, timeout: float = 2.0) -> None:
        """Drains what it can, then stops. Idempotent.

        Anything still buffered when `timeout` expires is counted as abandoned
        rather than waited on: shutdown of the agent must not be held hostage by
        an ingest service that is down.
        """
        if self._stopping.is_set():
            return
        self.flush(timeout)
        self._stopping.set()
        self._buffer.close()
        self._thread.join(timeout)
        leftover = len(self._buffer.drain()) + self._inflight
        if leftover:
            with self._lock:
                self._counts["abandoned"] += leftover

    def metrics(self) -> CaptureMetrics:
        with self._lock:
            counts = dict(self._counts)
            last_error = self._last_error
        return CaptureMetrics(
            dropped=self._buffer.dropped, pending=len(self._buffer), last_error=last_error, **counts
        )

    def __enter__(self) -> "CaptureClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- worker side (owns every network call) -----------------------------

    def _run(self) -> None:
        while not self._stopping.is_set():
            item = self._buffer.take(timeout=self._poll_interval)
            if item is None:
                continue
            with self._lock:
                self._inflight = 1
            try:
                self._deliver(item)
            finally:
                with self._lock:
                    self._inflight = 0

    def _deliver(self, trajectory: dict) -> None:
        attempt = 0
        while not self._stopping.is_set():
            try:
                self._transport.send(trajectory)
                self._bump("sent")
                return
            except TransportError as exc:
                if exc.permanent:
                    # Already quarantined at the far end; resending adds noise.
                    self._record_error(str(exc))
                    self._bump("rejected")
                    return
                reason = str(exc)
            except Exception as exc:  # a third-party transport must not kill the worker
                reason = repr(exc)
            attempt += 1
            self._record_error(reason)
            self._bump("retries")
            if attempt >= self.max_attempts:
                # Back to the head of the buffer: it is still the oldest, so if
                # the buffer is full it is also the right one to lose.
                self._buffer.put_front(trajectory)
                self._stopping.wait(self.retry_max_delay)
                return
            self._stopping.wait(self._backoff(attempt))

    def _backoff(self, attempt: int) -> float:
        return min(self.retry_base_delay * (2 ** (attempt - 1)), self.retry_max_delay)

    def _bump(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1

    def _record_error(self, reason: str) -> None:
        with self._lock:
            self._last_error = reason
