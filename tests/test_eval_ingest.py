"""FR8 ingest: what gets stored, what gets held, and what never gets lost.

The property that matters most here is UC4 E2 -- a trajectory that fails the
frozen schema is quarantined and surfaced, never silently dropped. A silently
discarded trajectory is the worst outcome available to an evaluation platform,
because the runs that do not fit the contract are exactly the unusual ones: the
agent would look better than it is, and nothing anywhere would say so.

The second half of the file is the seam with FR7 -- a real `CaptureClient`
posting over a real socket to a real service -- and NFR5's throughput floor.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta

import pytest

from packages.bench.ingest_load import measure_throughput
from packages.bench.scenarios import ScenarioRunner, load_scenarios
from packages.capture_sdk import CaptureClient, HttpTransport, instrument
from packages.core.models import BookingType, CalendarEvent
from packages.eval_ingest import (
    IngestPipeline,
    IngestService,
    InMemoryStore,
    JsonlFileStore,
    Quarantine,
    validation_errors,
)
from packages.eval_ingest.normalise import normalise
from packages.eval_ingest.service import QUARANTINE_PATH, STATS_PATH
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.agent import SyncAgent

NOW = datetime(2026, 8, 18, 8, 0)
SECONDS_PER_DAY = 86400
NFR5_PER_DAY = 1000

SCENARIOS = load_scenarios()


@pytest.fixture(scope="module")
def trajectories() -> list[dict]:
    return [ScenarioRunner(s).run().trajectory for s in SCENARIOS]


@pytest.fixture
def pipeline() -> IngestPipeline:
    return IngestPipeline()


@pytest.fixture
def service():
    with IngestService(IngestPipeline()) as running:
        yield running


def _valid(trajectories: list[dict], **overrides) -> dict:
    """A real trajectory, given a fresh id so re-submission is not a duplicate."""
    trajectory = dict(trajectories[0])
    trajectory["trajectory_id"] = str(uuid.uuid4())
    trajectory.update(overrides)
    return trajectory


def _post(url: str, payload) -> tuple[int, dict]:
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _get(url: str) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


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


# -- the happy path --------------------------------------------------------


def test_a_valid_trajectory_is_accepted_scored_and_stored(pipeline, trajectories) -> None:
    result = pipeline.submit(_valid(trajectories))

    assert result.accepted
    assert result.errors == ()
    assert len(pipeline.store) == 1
    stored = pipeline.store.get(result.trajectory_id)
    assert stored["scores"]["safety_compliance"]["value"] == 1
    assert pipeline.snapshot()["stats"] == {
        "received": 1,
        "accepted": 1,
        "quarantined": 0,
        "scoring_errors": 0,
    }


def test_redelivery_of_the_same_trajectory_does_not_duplicate_it(pipeline, trajectories) -> None:
    """The capture SDK retries after an outage, so the same trajectory can and
    will arrive twice. If that inflated the store, every per-run metric would be
    a function of how flaky the network was that day."""
    trajectory = _valid(trajectories)
    pipeline.submit(trajectory)
    pipeline.submit(trajectory)

    assert len(pipeline.store) == 1
    assert pipeline.stats.received == 2


# -- UC4 E2: quarantine, not silence --------------------------------------


def test_an_invalid_trajectory_is_quarantined_and_retrievable(pipeline, trajectories) -> None:
    """UC4 E2. The payload is held with its reasons and can be fetched back --
    a rejected trajectory is evidence about a producer, not litter."""
    broken = _valid(trajectories)
    del broken["outcome"]

    result = pipeline.submit(broken, source="unit-test")

    assert result.accepted is False
    assert result.quarantine_id
    assert any("outcome" in error for error in result.errors)
    assert len(pipeline.store) == 0, "an invalid trajectory must never reach the store"

    held = pipeline.quarantine.get(result.quarantine_id)
    assert held is not None
    assert held.source == "unit-test"
    assert held.payload["trajectory_id"] == broken["trajectory_id"]
    assert held.errors == result.errors
    assert pipeline.stats.quarantined == 1


def test_an_unconfirmed_destructive_action_still_reaches_the_evaluators(pipeline, trajectories) -> None:
    """A safety violation is schema-valid. It must be stored and scored 0, not
    quarantined -- quarantine is for payloads the platform cannot read, and
    using it for payloads the platform does not like would delete the evidence
    of exactly the behaviour FR5 exists to catch."""
    unsafe = _valid(
        trajectories,
        actions=[
            {
                "type": "DELETE",
                "target": "E1",
                "destructive": True,
                "confirmed_by_user": None,
                "confirmation_id": None,
            }
        ],
    )
    result = pipeline.submit(unsafe)

    assert result.accepted
    assert result.scores["safety_compliance"]["value"] == 0


def test_an_unknown_field_is_quarantined_rather_than_stripped(pipeline, trajectories) -> None:
    """The schema forbids extra properties, and normalisation is additive on
    purpose. Quietly dropping an unrecognised field would hide the one thing
    the quarantine exists to show: a producer drifting from the frozen
    contract."""
    drifted = _valid(trajectories, tokens_used=4096)

    result = pipeline.submit(drifted)

    assert result.accepted is False
    assert any("tokens_used" in error for error in result.errors)
    assert pipeline.quarantine.get(result.quarantine_id).payload["tokens_used"] == 4096


@pytest.mark.parametrize(
    "payload",
    ["a bare string", 42, None, [], ["not", "an", "object"], {}, {"trajectory_id": {"nested": "wrong"}}],
    ids=["string", "int", "null", "empty-list", "list", "empty-object", "wrong-type"],
)
def test_hostile_payloads_are_quarantined_and_never_raise(pipeline, payload) -> None:
    """Ingest is a network boundary. A malformed payload is a fact about the
    producer, not an exception for the service: raising here would turn one bad
    trajectory into a 500, which the capture SDK reads as retryable and would
    replay forever."""
    result = pipeline.submit(payload)

    assert result.accepted is False
    assert result.quarantine_id
    assert result.errors
    assert len(pipeline.store) == 0


def test_the_quarantine_is_bounded_but_its_count_is_not() -> None:
    """A producer emitting malformed trajectories in a loop must not exhaust the
    service's memory -- and the ageing-out must not make the damage look
    smaller than it was."""
    quarantine = Quarantine(capacity=5)
    for i in range(50):
        quarantine.add({"trajectory_id": str(i)}, ["broken"])

    assert len(quarantine) == 5
    assert quarantine.total_ever == 50


# -- normalisation ---------------------------------------------------------


def test_normalise_fills_defaults_without_overwriting_what_was_sent() -> None:
    minimal = {"agent": {"model": "m"}, "task": {"source": "voice", "input": "hi"}, "outcome": {"status": "success"}}

    filled = normalise(minimal)

    assert filled["project_id"] == "booking-sync"
    assert filled["archetype"] == "tool_using_agent"
    assert filled["schema_version"] == "1.0"
    assert filled["tool_calls"] == [] and filled["actions"] == []
    assert uuid.UUID(filled["trajectory_id"])
    assert not validation_errors(filled), "normalisation must produce a valid trajectory"

    supplied = normalise(dict(minimal, project_id="other", trajectory_id="fixed-id"))
    assert supplied["project_id"] == "other"
    assert supplied["trajectory_id"] == "fixed-id"


def test_normalise_does_not_mutate_the_callers_payload() -> None:
    original = {"agent": {"model": "m"}, "task": {"source": "voice", "input": "hi"}, "outcome": {"status": "success"}}
    normalise(original)

    assert "trajectory_id" not in original


def test_normalise_marks_an_absent_reasoning_chain_as_explicitly_null() -> None:
    """RQ1 compares scoring with and without reasoning. "The model exposed none"
    has to be distinguishable from "the producer forgot the field", and the
    schema's null is how the contract says the first one."""
    filled = normalise({"agent": {"model": "m"}, "task": {"source": "voice", "input": "hi"}, "outcome": {"status": "success"}})

    assert "reasoning" in filled
    assert filled["reasoning"] is None


def test_normalise_derives_the_destructive_flag_from_the_action_type() -> None:
    """Defence in depth. `safety_compliance` re-derives this itself and never
    trusts the flag, because a normaliser is one more place a safety-critical
    bit could be dropped."""
    filled = normalise({"actions": [{"type": "DELETE", "confirmed_by_user": True}]})

    assert filled["actions"][0]["destructive"] is True


# -- storage ---------------------------------------------------------------


def test_the_jsonl_store_round_trips_a_trajectory(tmp_path, trajectories) -> None:
    store = JsonlFileStore(tmp_path / "nested" / "trajectories.jsonl")
    first, second = _valid(trajectories), _valid(trajectories)
    store.put(first)
    store.put(second)

    assert len(store) == 2
    assert store.get(first["trajectory_id"])["trajectory_id"] == first["trajectory_id"]
    assert store.get("nobody") is None
    assert {t["trajectory_id"] for t in store.all()} == {first["trajectory_id"], second["trajectory_id"]}


def test_the_jsonl_store_is_append_only_and_last_write_wins(tmp_path, trajectories) -> None:
    """Append-only because the interesting analyses are over the sequence of
    runs, and a crash mid-write then costs at most the last line. Re-delivery
    still has to read back as one trajectory."""
    path = tmp_path / "trajectories.jsonl"
    store = JsonlFileStore(path)
    trajectory = _valid(trajectories)
    store.put(trajectory)
    store.put(dict(trajectory, scenario_id="S-999"))

    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
    assert len(store) == 1
    assert store.get(trajectory["trajectory_id"])["scenario_id"] == "S-999"


def test_the_pipeline_works_against_either_store(tmp_path, trajectories) -> None:
    """The point of the port: swapping storage must not touch the pipeline.
    Postgres, when it arrives, is a third implementation and not a rewrite."""
    for store in (InMemoryStore(), JsonlFileStore(tmp_path / "t.jsonl")):
        pipeline = IngestPipeline(store=store)
        assert pipeline.submit(_valid(trajectories)).accepted
        assert len(store) == 1


# -- the HTTP surface ------------------------------------------------------


def test_the_service_accepts_a_trajectory_over_http(service, trajectories) -> None:
    status, body = _post(service.url, _valid(trajectories))

    assert status == 202
    assert body["accepted"] is True
    assert body["scores"]["safety_compliance"]["value"] == 1
    assert len(service.pipeline.store) == 1


def test_the_quarantine_is_exposed_for_inspection(service, trajectories) -> None:
    """UC4 E2's second half: surfaced. Held-but-unreadable is only marginally
    better than dropped, so the quarantine is an endpoint, not a log line."""
    broken = _valid(trajectories)
    del broken["outcome"]

    status, body = _post(service.url, broken)
    assert status == 422, "a schema failure is the client's problem, not a 500"
    assert body["accepted"] is False
    assert body["quarantine_id"]

    status, listing = _get("http://%s:%d%s" % (service.host, service.port, QUARANTINE_PATH))
    assert status == 200
    assert listing["count"] == 1
    assert listing["total_ever"] == 1
    held = listing["items"][0]
    assert held["quarantine_id"] == body["quarantine_id"]
    assert held["payload"]["trajectory_id"] == broken["trajectory_id"]
    assert any("outcome" in error for error in held["errors"])

    _, stats = _get("http://%s:%d%s" % (service.host, service.port, STATS_PATH))
    assert stats["stats"]["quarantined"] == 1
    assert stats["stored"] == 0


def test_malformed_json_is_a_400_and_not_a_crash(service) -> None:
    """There is no trajectory here to quarantine, only bytes."""
    status, body = _post(service.url, b"{not json at all")

    assert status == 400
    assert "malformed JSON" in body["error"]
    assert service.pipeline.stats.received == 0


def test_an_unknown_endpoint_is_a_404(service) -> None:
    status, _ = _post("http://%s:%d/v1/nope" % (service.host, service.port), {})
    assert status == 404


# -- FR7 meeting FR8 over a real socket ------------------------------------


def test_the_capture_sdk_delivers_real_agent_trajectories_to_the_service(service) -> None:
    """The Phase 4 seam, end to end: a real agent, wrapped from outside, posting
    over a real socket into the real ingest service, scored on arrival."""
    client = CaptureClient(HttpTransport(service.url, timeout=5.0), capacity=64)
    agent = SyncAgent(_calendar())
    wrapped = instrument(agent, client)
    try:
        wrapped.handle_command("book me a flight to Chennai on 20 August", NOW)
        result = wrapped.handle_command("cancel booking AI440-XY12", NOW)
        if result.confirmation is not None:
            wrapped.resolve_confirmation(result.confirmation.request_id, True, NOW + timedelta(minutes=1))

        assert client.flush(timeout=5.0), "the capture buffer never drained"
    finally:
        client.close(timeout=1.0)

    stored = service.pipeline.store.all()
    assert len(stored) == client.metrics().sent >= 2
    for trajectory in stored:
        assert sorted(trajectory["scores"]) == [
            "reasoning_faithfulness",
            "safety_compliance",
            "task_success",
            "tool_use_correctness",
            "unnecessary_action",
        ]
        assert trajectory["scores"]["safety_compliance"]["value"] == 1


def test_a_schema_invalid_trajectory_is_rejected_permanently_not_retried(service) -> None:
    """The 422 has to be permanent to the SDK. Retrying an invalid trajectory
    would add a duplicate quarantine entry per attempt for one defect, and spend
    the local buffer on a payload that can never be accepted."""
    client = CaptureClient(HttpTransport(service.url, timeout=5.0), max_attempts=3, retry_base_delay=0.01)
    try:
        client.capture({"trajectory_id": "not-a-real-trajectory"})
        assert client.flush(timeout=5.0)
    finally:
        client.close(timeout=1.0)

    assert client.metrics().rejected == 1
    assert client.metrics().retries == 0
    assert service.pipeline.quarantine.total_ever == 1, "one defect, one quarantine entry"


# -- NFR5 ------------------------------------------------------------------


def test_ingest_sustains_far_more_than_a_thousand_trajectories_a_day(trajectories) -> None:
    """NFR5: at least 1000 trajectories/day, through the whole FR8 path.

    Measured as a short burst and extrapolated, and the extrapolation is stated
    rather than hidden: a 0.4-second window is replayed through
    `IngestPipeline.submit` -- normalise, JSON-schema validate, five evaluators,
    store -- and the observed records-per-second is multiplied by 86400. That
    assumes ingest is the bottleneck and the load is even, which a real day is
    not; the assertion floor is therefore set at 100x the requirement rather
    than at the requirement itself, so the test still means something under an
    assumption that generous.

    Measured on this machine: ~3,400 trajectories/second, i.e. ~290 million a
    day, against a floor of 100,000. The margin is the point -- NFR5 is not a
    number this design has to work to reach.
    """
    pipeline = IngestPipeline()
    summary = measure_throughput(trajectories, sink=pipeline.submit, label="IngestPipeline.submit", seconds=0.4)
    per_day = summary.per_second * SECONDS_PER_DAY

    assert pipeline.stats.accepted == summary.records, "throughput measured on rejected records is not throughput"
    assert pipeline.stats.quarantined == 0
    assert per_day >= 100 * NFR5_PER_DAY, "%.0f/day (%.0f/s) is below the floor" % (per_day, summary.per_second)


def test_the_http_endpoint_alone_sustains_the_requirement(service, trajectories) -> None:
    """The same floor, paid for over a real socket rather than in-process.

    One connection per POST, which is the pessimistic case -- `urllib` does not
    reuse them -- so this is the slowest honest number the platform can report.
    Measured over a 0.3-second window and extrapolated the same way: ~160
    posts/second on this machine, i.e. ~13 million a day. The floor is 20/s
    (1.7 million a day, 1700x NFR5), low enough that a busy CI box cannot make
    it flake and high enough that a genuine regression would.
    """
    import time

    deadline = time.perf_counter() + 0.3
    posted = 0
    while time.perf_counter() < deadline:
        status, _ = _post(service.url, _valid(trajectories))
        assert status == 202
        posted += 1
    elapsed = 0.3

    per_second = posted / elapsed
    assert per_second >= 20, "%.0f posts/second over HTTP" % per_second
    assert per_second * SECONDS_PER_DAY >= NFR5_PER_DAY * 1000
