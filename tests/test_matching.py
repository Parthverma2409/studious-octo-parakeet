"""Matching and decision unit tests (FR3, NFR2)."""

from __future__ import annotations

from datetime import datetime

from packages.core.decisions import ActionType, Reason
from packages.core.matching import match_booking
from packages.core.models import (
    BookingStatus,
    BookingType,
    CalendarEvent,
    FlightBooking,
    Source,
)
from packages.sync_agent.reconciliation import decide


def _cal(event_id: str, title: str, start: datetime, ref: str | None = None) -> CalendarEvent:
    return CalendarEvent(
        event_id=event_id,
        title=title,
        start_time=start,
        booking_ref=ref,
        booking_type=BookingType.FLIGHT,
    )


def _booking(start: datetime, ref: str | None = None, status: BookingStatus = BookingStatus.CONFIRMED) -> FlightBooking:
    return FlightBooking(
        booking_id="B1",
        booking_type=BookingType.FLIGHT,
        date_time=start,
        status=status,
        source=Source.EMAIL,
        title="Flight AI440 BLR to MAA",
        booking_ref=ref,
        destination="Chennai",
    )


def test_reference_match_beats_everything() -> None:
    events = [
        _cal("E1", "Something else entirely", datetime(2026, 8, 20, 9, 15), ref="AI440-XY12"),
        _cal("E2", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15)),
    ]
    result = match_booking(events, _booking(datetime(2026, 8, 20, 9, 15), ref="AI440-XY12"))
    assert result.by_reference
    assert result.matched is not None and result.matched.event_id == "E1"


def test_two_equally_plausible_events_are_ambiguous_not_guessed() -> None:
    events = [
        _cal("E1", "Flight to Chennai morning", datetime(2026, 8, 20, 9, 15)),
        _cal("E2", "Flight to Chennai evening", datetime(2026, 8, 20, 19, 15)),
    ]
    result = match_booking(events, _booking(datetime(2026, 8, 20, 12, 0)))
    assert result.matched is None
    assert result.is_ambiguous


def test_events_far_apart_are_separate_bookings() -> None:
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15))]
    result = match_booking(events, _booking(datetime(2026, 9, 15, 9, 15)))
    assert result.matched is None
    assert not result.is_ambiguous


def test_identical_notification_is_a_noop() -> None:
    """NFR2 in its smallest form."""
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15), ref="AI440-XY12")]
    decision = decide(events, _booking(datetime(2026, 8, 20, 9, 15), ref="AI440-XY12"))
    assert decision.action is ActionType.NOOP
    assert decision.reason is Reason.DUPLICATE_ALREADY_APPLIED


def test_time_change_is_an_update_and_is_destructive() -> None:
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15), ref="AI440-XY12")]
    decision = decide(events, _booking(datetime(2026, 8, 20, 14, 45), ref="AI440-XY12"))
    assert decision.action is ActionType.UPDATE
    assert decision.requires_confirmation


def test_decide_is_pure() -> None:
    """Calling twice on the same inputs gives the same answer and mutates nothing."""
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15), ref="AI440-XY12")]
    booking = _booking(datetime(2026, 8, 20, 14, 45), ref="AI440-XY12")
    first, second = decide(events, booking), decide(events, booking)
    assert first.action is second.action and first.reason is second.reason
    assert events[0].start_time == datetime(2026, 8, 20, 9, 15)


def test_weak_title_overlap_does_not_match() -> None:
    """Pins the fuzzy threshold from below.

    Two flights on one day sharing only the connecting airport are separate legs of
    an itinerary. Matching them would silently merge a two-leg trip into one event.
    """
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15))]
    booking = FlightBooking(
        booking_id="B2",
        booking_type=BookingType.FLIGHT,
        date_time=datetime(2026, 8, 20, 18, 40),
        status=BookingStatus.CONFIRMED,
        source=Source.EMAIL,
        title="Flight 6E211 MAA to DEL",
    )
    assert match_booking(events, booking).matched is None


def test_strong_title_overlap_matches() -> None:
    """Pins the same threshold from above, so a regression in either direction fails."""
    events = [_cal("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15))]
    booking = FlightBooking(
        booking_id="B3",
        booking_type=BookingType.FLIGHT,
        date_time=datetime(2026, 8, 20, 14, 45),
        status=BookingStatus.CONFIRMED,
        source=Source.EMAIL,
        title="Flight AI440 BLR to MAA",
    )
    result = match_booking(events, booking)
    assert result.matched is not None and result.matched.event_id == "E1"


def test_new_booking_carries_an_alarm() -> None:
    decision = decide([], _booking(datetime(2026, 8, 20, 9, 15), ref="AI440-XY12"))
    assert decision.action is ActionType.CREATE
    assert decision.payload is not None and decision.payload.alarm is not None
    assert decision.payload.alarm.trigger_time == datetime(2026, 8, 20, 8, 15)


def test_reschedule_preserves_the_existing_title_and_detail() -> None:
    """A time change must not silently rename the user's calendar entry.

    Change notices routinely carry less detail than the original confirmation,
    so taking their title wholesale degrades the entry the user recognises.
    """
    events = [
        CalendarEvent(
            event_id="E1",
            title="Flight AI440 Bengaluru (BLR) to Chennai (MAA)",
            start_time=datetime(2026, 8, 20, 9, 15),
            end_time=datetime(2026, 8, 20, 12, 0),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
            location="BLR",
        )
    ]
    sparse = FlightBooking(
        booking_id="B9",
        booking_type=BookingType.FLIGHT,
        date_time=datetime(2026, 8, 20, 14, 45),
        status=BookingStatus.RESCHEDULED,
        source=Source.EMAIL,
        title="Flight AI440-XY12",
        booking_ref="AI440-XY12",
    )
    decision = decide(events, sparse)

    assert decision.action is ActionType.UPDATE
    assert decision.payload.start_time == datetime(2026, 8, 20, 14, 45)
    assert decision.payload.title == "Flight AI440 Bengaluru (BLR) to Chennai (MAA)"
    # Detail the change notice simply did not mention survives too.
    assert decision.payload.location == "BLR"
    assert decision.payload.end_time == datetime(2026, 8, 20, 12, 0)
