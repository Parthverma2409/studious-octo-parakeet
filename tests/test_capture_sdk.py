"""FR7 capture: the assistant must not pay for the evaluation platform.

The property under test throughout this file is UC4 E1 -- when the ingest
service is unreachable, the Sync Agent's user-facing response is neither blocked
nor slowed. It is measured, not asserted by construction: a design can be
correct on paper and still put a socket call on the caller's thread through one
careless refactor, and the only check that catches that is a stopwatch.

The rest of the file covers what the buffer is allowed to lose and how loudly it
says so, because "best effort" is only acceptable when the effort is counted.
"""

from __future__ import annotations

import socket
import statistics
import threading
import time
from datetime import datetime, timedelta

import pytest
from jsonschema import Draft202012Validator

from packages.capture_sdk import (
    BoundedBuffer,
    CaptureClient,
    HttpTransport,
    MemoryTransport,
    TransportError,
    instrument,
)
from packages.core.models import BookingType, CalendarEvent
from packages.providers.fake_calendar import FakeCalendar
from packages.providers.mail import RawEmail
from packages.sync_agent.agent import SyncAgent
from tests.test_trajectory_contract import SCHEMA

NOW = datetime(2026, 8, 18, 8, 0)
BOOK = "book me a flight to Chennai on 20 August"


# -- doubles ---------------------------------------------------------------


class HangingTransport:
    """Never returns. Stands in for an ingest service that accepts the
    connection and then stops responding -- the worst case for a caller that
    waits, and a no-op for one that does not."""

    def __init__(self) -> None:
        self.released = threading.Event()
        self.entered = threading.Event()

    def send(self, trajectory: dict) -> None:
        self.entered.set()
        self.released.wait(30)


class FlakyTransport:
    """Fails `failures` times, then works. An outage that ends."""

    def __init__(self, failures: int, permanent: bool = False):
        self.remaining = failures
        self.permanent = permanent
        self.attempts = 0
        self.received: list[dict] = []
        self._lock = threading.Lock()

    def send(self, trajectory: dict) -> None:
        with self._lock:
            self.attempts += 1
            if self.remaining > 0:
                self.remaining -= 1
                raise TransportError("ingest unreachable", permanent=self.permanent)
            self.received.append(trajectory)


class ExplodingTransport:
    """Raises something the SDK has never heard of. A third-party transport is
    not obliged to honour `TransportError`, and must not be able to kill the
    worker thread by getting that wrong.

    `failures=None` means it never recovers.
    """

    def __init__(self, failures: int | None = None):
        self.remaining = failures
        self.attempts = 0
        self.received: list[dict] = []
        self._lock = threading.Lock()

    def send(self, trajectory: dict) -> None:
        with self._lock:
            self.attempts += 1
            if self.remaining is None or self.remaining > 0:
                if self.remaining is not None:
                    self.remaining -= 1
                raise ZeroDivisionError("a transport nobody tested")
            self.received.append(trajectory)


class BrokenClient:
    """A capture client that raises on every call."""

    def capture(self, trajectory: dict) -> bool:
        raise RuntimeError("capture backend is on fire")


@pytest.fixture
def clients():
    """Closes every client a test made, including on failure -- a leaked worker
    thread turns one failing test into a confusing suite-wide hang."""
    made: list[CaptureClient] = []
    yield made
    for client in made:
        client.close(timeout=0.2)


def _client(transport, clients, **kwargs) -> CaptureClient:
    defaults = dict(capacity=64, max_attempts=2, retry_base_delay=0.005, retry_max_delay=0.02)
    defaults.update(kwargs)
    client = CaptureClient(transport, **defaults)
    clients.append(client)
    return client


def _calendar() -> FakeCalendar:
    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            event_id="E1",
            title="Flight AI440 BLR to MAA",
            start_time=datetime(2026, 8, 20, 9, 15),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
        ).with_alarm_offset(60, "AL-E1")
    )
    return calendar


def _closed_port() -> int:
    """A port nothing is listening on: bind it, read it, release it.

    Better than a hard-coded number, which is either occupied by a real service
    on somebody's machine or, worse, occupied by a previous run of this suite.
    """
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _median_latency_ms(agent, samples: int) -> float:
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        agent.handle_command(BOOK, NOW)
        timings.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(timings)


# -- UC4 E1: the headline property ----------------------------------------


def test_agent_latency_is_unchanged_when_the_ingest_service_is_unreachable(clients) -> None:
    """FR7's whole point, measured: an ingest outage costs the user nothing.

    Two real `SyncAgent`s answer the same command 300 times each; one of them is
    instrumented with a capture client pointed at a port nothing is listening
    on, so every delivery attempt fails for real over a real socket. Medians are
    compared rather than means, because one Windows scheduler hiccup in 300
    samples should not decide a correctness question.

    The bound is 0.5 ms of added median latency, against a measured delta of
    roughly +/-0.01 ms -- capture is a lock and a deque append. The margin is
    deliberately fifty times the observed effect: this test is here to catch a
    refactor that puts the socket back on the caller's thread, which would cost
    milliseconds at least, not to police microseconds of scheduler noise.
    """
    baseline = _median_latency_ms(SyncAgent(_calendar()), 300)

    url = "http://127.0.0.1:%d/v1/trajectories" % _closed_port()
    client = _client(HttpTransport(url, timeout=0.25), clients, capacity=64)
    instrumented = _median_latency_ms(instrument(SyncAgent(_calendar()), client), 300)

    metrics = client.metrics()
    assert metrics.captured == 300, "the test proved nothing if capture never ran"
    assert metrics.sent == 0, "the service was supposed to be unreachable"
    assert instrumented - baseline < 0.5, (
        "capture added %.4f ms of median latency (baseline %.4f ms, instrumented %.4f ms)"
        % (instrumented - baseline, baseline, instrumented)
    )


def test_capture_returns_while_the_transport_is_still_blocked(clients) -> None:
    """The same property without a stopwatch: a transport that never returns.

    If any part of delivery ran on the caller's thread this test would hang for
    30 seconds rather than fail, which is its own kind of signal.
    """
    transport = HangingTransport()
    client = _client(transport, clients, capacity=256)

    start = time.perf_counter()
    for _ in range(200):
        client.capture({"trajectory_id": "t", "actions": []})
    elapsed = time.perf_counter() - start

    assert transport.entered.wait(2.0), "the worker never even started sending"
    assert elapsed < 0.1, "200 captures took %.3fs while the transport was blocked" % elapsed
    assert client.metrics().sent == 0
    transport.released.set()


def test_capture_does_not_raise_when_the_transport_is_broken(clients) -> None:
    """A capture path that can raise is a capture path that can fail a booking."""
    client = _client(ExplodingTransport(), clients)
    assert client.capture({"trajectory_id": "t"}) is True
    client.flush(timeout=0.2)
    assert client.metrics().sent == 0


# -- the bounded buffer ----------------------------------------------------


def test_the_buffer_drops_the_oldest_and_keeps_the_newest() -> None:
    """Bounded means bounded, and the survivors are the recent ones."""
    buffer = BoundedBuffer(capacity=3)
    for i in range(10):
        buffer.put(i)

    assert len(buffer) == 3
    assert buffer.dropped == 7
    assert [buffer.take(0.01) for _ in range(3)] == [7, 8, 9]


def test_a_returned_item_is_dropped_rather_than_displacing_newer_work() -> None:
    """`put_front` is the retry path. When the buffer is full the item coming
    back is the oldest one there is, so it is the right one to lose."""
    buffer = BoundedBuffer(capacity=2)
    buffer.put("a")
    buffer.put("b")

    assert buffer.put_front("older") is False
    assert buffer.dropped == 1
    assert [buffer.take(0.01) for _ in range(2)] == ["a", "b"]


def test_an_ingest_outage_bounds_memory_and_counts_the_loss(clients) -> None:
    """The failure this prevents: an evaluation outage OOM-killing the assistant."""
    client = _client(HangingTransport(), clients, capacity=8)
    for i in range(200):
        client.capture({"trajectory_id": str(i)})

    metrics = client.metrics()
    assert metrics.pending <= 8
    assert metrics.dropped >= 190
    # Nothing is lost silently: every captured trajectory is either pending,
    # delivered, dropped or in flight, and all of those are counted.
    assert metrics.captured == 200
    assert metrics.dropped + metrics.pending + metrics.sent >= 199


# -- retry, rejection, shutdown -------------------------------------------


def test_a_trajectory_survives_an_outage_that_ends(clients) -> None:
    """UC4 E1's second half: buffer locally and retry. Two failures, then the
    service comes back, and the trajectory arrives without the caller knowing
    any of it happened."""
    transport = FlakyTransport(failures=2)
    client = _client(transport, clients)

    client.capture({"trajectory_id": "t1"})

    assert client.flush(timeout=3.0), "the trajectory never drained"
    assert [t["trajectory_id"] for t in transport.received] == ["t1"]
    assert transport.attempts == 3
    assert client.metrics().retries == 2


def test_a_rejected_trajectory_is_not_retried_forever(clients) -> None:
    """A schema-invalid trajectory does not become valid on the third attempt.

    The far end has already quarantined it (UC4 E2), so retrying would add
    duplicate quarantine entries for one defect and spend the buffer on a
    payload that can never be accepted.
    """
    transport = FlakyTransport(failures=5, permanent=True)
    client = _client(transport, clients)

    client.capture({"trajectory_id": "bad"})
    client.flush(timeout=1.0)

    assert transport.attempts == 1
    assert client.metrics().rejected == 1
    assert client.metrics().retries == 0


def test_flush_reports_failure_rather_than_pretending(clients) -> None:
    """`flush` returning True is a claim that the record is complete. With the
    service down it has to be False, or a shutdown path would report success
    while dropping the buffer on the floor."""
    client = _client(HangingTransport(), clients)
    client.capture({"trajectory_id": "t"})

    assert client.flush(timeout=0.1) is False
    assert client.metrics().pending + 1 >= 1


def test_close_is_idempotent_and_refuses_late_captures(clients) -> None:
    transport = MemoryTransport()
    client = _client(transport, clients)
    client.capture({"trajectory_id": "t1"})
    client.close(timeout=1.0)
    client.close(timeout=1.0)

    assert len(transport) == 1
    assert client.capture({"trajectory_id": "t2"}) is False
    assert client.metrics().refused == 1


def test_close_counts_what_it_abandons(clients) -> None:
    """Shutdown is not allowed to hang on a dead ingest service, so it drops the
    remainder -- and says how much."""
    client = _client(HangingTransport(), clients, capacity=16)
    for i in range(5):
        client.capture({"trajectory_id": str(i)})
    client.close(timeout=0.1)

    metrics = client.metrics()
    assert metrics.sent == 0
    assert metrics.abandoned >= 1
    assert metrics.lost >= 1


def test_the_worker_survives_a_transport_that_raises_anything(clients) -> None:
    """One unexpected exception must not end capture for the process lifetime.

    A transport is third-party code and is under no obligation to raise
    `TransportError`. If an unrecognised exception escaped the delivery loop the
    worker thread would die silently, capture would stop for the life of the
    process, and the first symptom would be an empty dashboard days later.
    """
    transport = ExplodingTransport(failures=2)
    client = _client(transport, clients)
    client.capture({"trajectory_id": "t1"})
    client.capture({"trajectory_id": "t2"})

    assert client.flush(timeout=3.0), "the worker stopped after an unknown exception"
    assert transport.attempts >= 3
    assert sorted(t["trajectory_id"] for t in transport.received) == ["t1", "t2"]


# -- instrumentation -------------------------------------------------------


def test_instrument_captures_every_entry_point_without_editing_the_agent(clients) -> None:
    """FR7 wiring is external. All three entry points are captured, and each
    trajectory is the schema-valid one the agent already emits."""
    transport = MemoryTransport()
    client = _client(transport, clients)
    agent = SyncAgent(_calendar())
    wrapped = instrument(agent, client)

    wrapped.handle_command(BOOK, NOW)
    email = RawEmail(
        message_id="m1",
        sender="airline@example.com",
        subject="Cancelled: your flight AI440 on 20 August",
        body="Your booking AI440-XY12 on 20 August at 09:15 has been cancelled.",
        received_at=NOW,
    )
    pending = wrapped.handle_email(email, NOW)
    if pending.confirmation is not None:
        wrapped.resolve_confirmation(pending.confirmation.request_id, True, NOW + timedelta(minutes=1))

    assert client.flush(timeout=2.0)
    assert len(transport.received) >= 2
    validator = Draft202012Validator(SCHEMA)
    for trajectory in transport.received:
        assert not list(validator.iter_errors(trajectory)), trajectory


def test_part_a_never_imports_part_b() -> None:
    """NFR6, checked structurally rather than promised in a docstring.

    Part A must still run, and still pass its tests, with the evaluation
    platform deleted from the repo. The moment anything under `sync_agent/`,
    `core/`, `nlp/` or `providers/` imports the capture SDK or the ingest
    service, the assistant has taken a dependency on its own observer and this
    whole module's premise is gone. Instrumentation is a wrapper, from outside,
    or it is not non-invasive.
    """
    import ast
    from pathlib import Path

    part_a = Path(__file__).resolve().parents[1] / "packages"
    offenders = []
    for directory in ("sync_agent", "core", "nlp", "providers"):
        for path in sorted((part_a / directory).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                if any("capture_sdk" in n or "eval_ingest" in n for n in names):
                    offenders.append("%s:%d" % (path.name, node.lineno))

    assert not offenders, "Part A imports Part B at %s" % offenders


def test_the_proxy_returns_the_agent_result_untouched(clients) -> None:
    """Instrumentation observes; it does not get a vote in what the user sees."""
    client = _client(MemoryTransport(), clients)
    agent = SyncAgent(_calendar())
    wrapped = instrument(agent, client)

    direct = SyncAgent(_calendar()).handle_command(BOOK, NOW)
    through_proxy = wrapped.handle_command(BOOK, NOW)

    assert through_proxy.status == direct.status
    assert through_proxy.message == direct.message
    # Anything not on the captured list is the agent's own attribute, not a copy.
    assert wrapped.calendar is agent.calendar
    assert wrapped.confirmations is agent.confirmations
    assert wrapped.agent is agent


def test_an_agent_exception_propagates_and_captures_nothing(clients) -> None:
    """A half-built trajectory is worse than no trajectory: it would be scored."""
    transport = MemoryTransport()
    client = _client(transport, clients)

    class Failing:
        def handle_command(self, text: str, now: datetime) -> None:
            raise ValueError("extractor exploded")

    wrapped = instrument(Failing(), client)
    with pytest.raises(ValueError, match="extractor exploded"):
        wrapped.handle_command(BOOK, NOW)

    client.flush(timeout=0.2)
    assert transport.received == []


def test_a_broken_capture_client_cannot_break_a_booking() -> None:
    """The agent completed the user's task. An evaluation backend that is down
    must not turn that into an error the user sees."""
    agent = SyncAgent(_calendar())
    wrapped = instrument(agent, BrokenClient())

    result = wrapped.handle_command(BOOK, NOW)

    assert result.status == "success"
    assert len(agent.calendar.events) == 2
