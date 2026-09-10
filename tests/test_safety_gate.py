"""Unit tests for the FR5 capability gate itself.

The scenario suite proves the agent behaves. These prove the gate cannot be
circumvented even by code that tries -- which is what makes FR5 a property of the
design rather than a convention future contributors must remember.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from packages.core.confirmation import ConfirmationStore
from packages.core.decisions import ActionType, ConfirmedDecision, Decision, Reason
from packages.core.models import BookingType, CalendarEvent
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.reconciliation import ReconciliationEngine

NOW = datetime(2026, 8, 18, 8, 0)


def _event() -> CalendarEvent:
    return CalendarEvent(
        event_id="E1",
        title="Flight AI440",
        start_time=datetime(2026, 8, 20, 9, 15),
        booking_ref="AI440-XY12",
        booking_type=BookingType.FLIGHT,
    )


def _delete_decision() -> Decision:
    return Decision(
        action=ActionType.DELETE,
        reason=Reason.CANCELLATION,
        target_event_id="E1",
        payload=_event(),
    )


def test_confirmed_decision_cannot_be_forged() -> None:
    with pytest.raises(PermissionError):
        ConfirmedDecision(_delete_decision(), "CR-fake", NOW)


def test_apply_refuses_destructive_decisions() -> None:
    calendar = FakeCalendar()
    calendar.seed(_event())
    engine = ReconciliationEngine(calendar)

    with pytest.raises(PermissionError):
        engine.apply(_delete_decision())

    assert calendar.mutations == []
    assert "E1" in calendar.events


def test_confirmed_deletion_goes_through() -> None:
    calendar = FakeCalendar()
    calendar.seed(_event())
    engine = ReconciliationEngine(calendar)
    store = ConfirmationStore()

    req = store.open(_delete_decision(), NOW)
    token = req.confirm(NOW)
    engine.apply_destructive(token)

    assert calendar.events == {}
    assert calendar.mutations == [("DELETE", "E1")]


def test_declined_confirmation_yields_no_token() -> None:
    store = ConfirmationStore()
    req = store.open(_delete_decision(), NOW)
    assert req.resolve(False, NOW) is None


def test_unintelligible_response_is_not_consent() -> None:
    """UC3 open issue: ambiguity must never resolve to yes."""
    store = ConfirmationStore()
    req = store.open(_delete_decision(), NOW)
    assert req.resolve(None, NOW) is None


def test_timed_out_confirmation_cannot_be_confirmed_later() -> None:
    store = ConfirmationStore()
    req = store.open(_delete_decision(), NOW, timeout=timedelta(minutes=15))
    with pytest.raises(TimeoutError):
        req.confirm(NOW + timedelta(minutes=30))


def test_non_destructive_decision_cannot_open_a_confirmation() -> None:
    store = ConfirmationStore()
    create = Decision(action=ActionType.CREATE, reason=Reason.NEW_BOOKING, payload=_event())
    with pytest.raises(ValueError):
        store.open(create, NOW)


def test_confirmation_prompt_names_the_event() -> None:
    """SRS UI priority 1: the prompt must be unambiguous."""
    prompt = _delete_decision().describe()
    assert "Flight AI440" in prompt
    assert "Delete" in prompt
