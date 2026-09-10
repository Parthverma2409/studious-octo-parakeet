"""FR6 -- spoken status queries.

This path had no test at all until an audit found `_answer_query` was called but
never defined: every query crashed with AttributeError while 282 tests passed.
The structural check in `test_repo_consistency.py` now catches that class of bug;
these tests cover the behaviour.

The property that matters most here is that a question is not an instruction.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from packages.core.models import BookingType, CalendarEvent
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.agent import SyncAgent

NOW = datetime(2026, 8, 18, 8, 0)


def _calendar() -> FakeCalendar:
    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            event_id="E1",
            title="my Chennai flight",
            start_time=datetime(2026, 8, 20, 9, 15),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
        )
    )
    calendar.seed(
        CalendarEvent(
            event_id="E2",
            title="the dentist appointment",
            start_time=datetime(2026, 8, 22, 17, 30),
            booking_type=BookingType.APPOINTMENT,
        )
    )
    return calendar


@pytest.mark.parametrize(
    "question",
    [
        "when is my Chennai flight",
        "what's the status of AI440-XY12",
        "is my Chennai flight still confirmed",
        "tell me about my Chennai flight",
        "do I have anything booked on 20 August",
        "when is the dentist appointment",
    ],
)
def test_queries_are_answered_without_touching_the_calendar(question: str) -> None:
    """A question must never mutate state or raise a confirmation."""
    calendar = _calendar()
    agent = SyncAgent(calendar)
    result = agent.handle_command(question, NOW)

    assert result.status == "success"
    assert calendar.mutations == []
    assert agent.confirmations.count() == 0


def test_query_reports_the_matching_booking() -> None:
    agent = SyncAgent(_calendar())
    result = agent.handle_command("when is my Chennai flight", NOW)
    assert "Chennai" in result.message
    assert "20 Aug" in result.message


def test_query_by_reference_finds_the_right_event() -> None:
    agent = SyncAgent(_calendar())
    result = agent.handle_command("what's the status of AI440-XY12", NOW)
    assert "Chennai" in result.message
    assert "dentist" not in result.message


def test_query_by_date_lists_that_day() -> None:
    agent = SyncAgent(_calendar())
    result = agent.handle_command("do I have anything booked on 22 August", NOW)
    # Asserting the ABSENCE matters as much as the presence: an earlier version of
    # this test passed by falling through to the "here is everything" branch,
    # which happened to contain the word it was looking for.
    assert "dentist" in result.message
    assert "Chennai" not in result.message


def test_query_for_a_free_day_says_so() -> None:
    agent = SyncAgent(_calendar())
    result = agent.handle_command("do I have anything booked on 25 August", NOW)
    assert "Nothing is booked" in result.message


def test_query_against_an_empty_calendar() -> None:
    agent = SyncAgent(FakeCalendar())
    result = agent.handle_command("when is my Chennai flight", NOW)
    assert result.status == "success"
    assert "nothing booked" in result.message.lower()


def test_unmatched_query_says_so_and_still_lists_what_exists() -> None:
    agent = SyncAgent(_calendar())
    result = agent.handle_command("when is my hotel booking", NOW)
    assert "could not find" in result.message.lower()
    assert "Chennai" in result.message


def test_a_query_emits_a_trajectory_with_no_actions() -> None:
    """FR7: a read-only interaction is still observable, and records no action."""
    agent = SyncAgent(_calendar())
    result = agent.handle_command("when is my Chennai flight", NOW)
    assert result.trajectory["actions"] == []
    assert result.trajectory["outcome"]["status"] == "success"
