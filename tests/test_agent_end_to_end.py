"""Phase 3 exit criterion: the design document's sequence diagram, from raw text.

Scenario S-401 already proves the reconciler handles a cancellation, but it is
fed *structured* input. These tests start from an actual email body and an actual
spoken sentence, so the NLP layer, the resolver, the reconciler and the FR5 gate
are all in the path.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from packages.core.models import BookingType, CalendarEvent
from packages.providers.fake_calendar import FakeCalendar
from packages.providers.fake_inbox import FakeInbox
from packages.sync_agent.agent import SyncAgent
from packages.sync_agent.email_watcher import EmailIngestionModule
from tests.test_trajectory_contract import SCHEMA

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "emails" / "fixtures"
NOW = datetime(2026, 8, 18, 8, 0)


def _calendar_with_flight() -> FakeCalendar:
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


def _agent(calendar: FakeCalendar) -> SyncAgent:
    return SyncAgent(calendar)


def test_sequence_diagram_end_to_end_from_a_real_email() -> None:
    """Email arrives -> prompt raised -> user confirms -> event and alarm gone."""
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    results = watcher.scan_inbox(NOW)

    assert len(results) == 1
    prompt = results[0]
    assert prompt.status == "awaiting_confirmation"
    assert "AI440" in prompt.message
    # Nothing has been touched yet -- this is the whole point of FR5.
    assert calendar.mutations == []
    assert "E1" in calendar.events

    confirmed = agent.resolve_confirmation(prompt.confirmation.request_id, True, NOW + timedelta(minutes=3))

    assert confirmed.status == "success"
    assert calendar.events == {}
    assert calendar.mutations == [("DELETE", "E1")]
    assert watcher.last_checked_at == NOW


def test_the_same_flow_declined_changes_nothing() -> None:
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")

    prompt = watcher.scan_inbox(NOW)[0]
    declined = agent.resolve_confirmation(prompt.confirmation.request_id, False, NOW + timedelta(minutes=3))

    assert declined.status == "refused"
    assert calendar.mutations == []
    assert "E1" in calendar.events


def test_promotional_email_is_dropped_before_it_reaches_the_calendar() -> None:
    """UC2's open issue, end to end. The subject line says "cancelled"."""
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    inbox.deliver_fixture(FIXTURES / "promo_fare_sale.eml")
    result = watcher.scan_inbox(NOW)[0]

    assert result.status == "refused"
    assert agent.confirmations.count() == 0
    assert calendar.mutations == []


def test_unparseable_email_is_quarantined_not_dropped() -> None:
    """UC2 E1: flagged for manual review, calendar untouched."""
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    inbox.deliver_fixture(FIXTURES / "unparseable_itinerary.eml")
    result = watcher.scan_inbox(NOW)[0]

    assert result.status == "failure"
    assert len(agent.quarantine) == 1
    assert calendar.mutations == []


def test_reschedule_email_is_gated_then_applied() -> None:
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    inbox.deliver_fixture(FIXTURES / "reschedule_ai440.eml")
    prompt = watcher.scan_inbox(NOW)[0]
    assert prompt.status == "awaiting_confirmation"
    assert calendar.mutations == []

    agent.resolve_confirmation(prompt.confirmation.request_id, True, NOW + timedelta(minutes=1))
    assert calendar.events["E1"].start_time == datetime(2026, 8, 20, 14, 45)


def test_duplicate_confirmation_email_writes_once() -> None:
    """NFR2 through the real email path, not just the structured one."""
    calendar = FakeCalendar()
    agent = _agent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    inbox.deliver_fixture(FIXTURES / "confirm_ai440.eml", message_id="m1")
    watcher.scan_inbox(NOW)
    inbox.deliver_fixture(FIXTURES / "confirm_ai440.eml", message_id="m2")
    watcher.scan_inbox(NOW + timedelta(minutes=30))

    assert len([m for m in calendar.mutations if m[0] == "CREATE"]) == 1


# -- spoken commands (UC1) ------------------------------------------------


def test_spoken_booking_creates_an_event() -> None:
    calendar = FakeCalendar()
    agent = _agent(calendar)
    result = agent.handle_command("book me a flight to Chennai on 20 August", NOW)

    assert result.status == "success"
    assert len(calendar.events) == 1
    event = next(iter(calendar.events.values()))
    assert event.start_time.date() == datetime(2026, 8, 20).date()
    assert event.alarm is not None


def test_spoken_cancellation_finds_the_event_without_a_date() -> None:
    """"Cancel my Chennai flight" carries no date; the calendar supplies it."""
    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            event_id="E1",
            title="my Chennai flight",
            start_time=datetime(2026, 8, 20, 9, 15),
            booking_type=BookingType.FLIGHT,
        )
    )
    agent = _agent(calendar)
    result = agent.handle_command("cancel my Chennai flight", NOW)

    assert result.status == "awaiting_confirmation"
    assert calendar.mutations == []

    agent.resolve_confirmation(result.confirmation.request_id, True, NOW)
    assert calendar.events == {}


def test_a_command_without_a_date_asks_instead_of_guessing() -> None:
    """UC1 E2. A guessed date is a missed flight."""
    calendar = FakeCalendar()
    agent = _agent(calendar)
    result = agent.handle_command("book me a flight to Chennai", NOW)

    assert result.needs_clarification
    assert calendar.mutations == []
    assert "date" in result.message.lower()


def test_out_of_scope_command_changes_nothing() -> None:
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    result = agent.handle_command("what's the weather in Chennai", NOW)

    assert result.status == "refused"
    assert calendar.mutations == []


def test_a_stray_yes_finds_nothing_to_confirm() -> None:
    """The most dangerous input in the system: consent with no question behind it."""
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    result = agent.handle_command("yes go ahead", NOW)

    assert result.status == "refused"
    assert calendar.mutations == []


def test_spoken_confirmation_resolves_the_pending_request() -> None:
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW)

    result = agent.handle_command("yes go ahead", NOW + timedelta(minutes=2))
    assert result.status == "success"
    assert calendar.events == {}


def test_a_self_correction_never_confirms() -> None:
    calendar = _calendar_with_flight()
    agent = _agent(calendar)
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW)

    result = agent.handle_command("no wait, don't cancel it", NOW + timedelta(minutes=2))
    assert result.status == "refused"
    assert "E1" in calendar.events


def test_ambiguous_spoken_reference_asks_rather_than_guessing() -> None:
    calendar = FakeCalendar()
    for event_id, hour in (("E1", 9), ("E2", 19)):
        calendar.seed(
            CalendarEvent(
                event_id=event_id,
                title="my Chennai flight",
                start_time=datetime(2026, 8, 20, hour, 15),
                booking_type=BookingType.FLIGHT,
            )
        )
    agent = _agent(calendar)
    result = agent.handle_command("cancel my Chennai flight", NOW)

    assert result.needs_clarification
    assert calendar.mutations == []
    assert agent.confirmations.count() == 0


@pytest.mark.parametrize(
    "text",
    [
        "book me a flight to Chennai on 20 August",
        "what's the weather in Chennai",
        "yes go ahead",
        "cancel booking AI440-XY12",
    ],
)
def test_every_interaction_emits_a_schema_valid_trajectory(text: str) -> None:
    """FR7: the agent already speaks the frozen schema, before the capture SDK exists."""
    agent = _agent(_calendar_with_flight())
    result = agent.handle_command(text, NOW)
    errors = list(Draft202012Validator(SCHEMA).iter_errors(result.trajectory))
    assert not errors, [e.message for e in errors]


def test_email_interactions_also_emit_valid_trajectories() -> None:
    agent = _agent(_calendar_with_flight())
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    for result in EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW):
        errors = list(Draft202012Validator(SCHEMA).iter_errors(result.trajectory))
        assert not errors, [e.message for e in errors]
