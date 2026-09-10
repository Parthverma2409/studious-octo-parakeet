"""ReconciliationEngine (FR3, FR4).

``decide`` is a pure function of (calendar state, incoming booking) -> Decision.
No network, no clock, no LLM. Everything that makes reconciliation hard to test
lives in ``apply`` / ``apply_destructive`` instead, which is why the 60-scenario
suite can run thousands of times for free and why the FR5 gate is provable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from packages.core.decisions import ActionType, ConfirmedDecision, Decision, Reason
from packages.core.matching import match_booking
from packages.core.models import (
    BookingEvent,
    BookingStatus,
    CalendarEvent,
)
from packages.providers.calendar import CalendarProvider

#: Two starts closer than this are "the same time" — vendors round and re-send.
SAME_TIME_TOLERANCE_SEC = 60


def _same_time(a: datetime, b: datetime) -> bool:
    return abs((a - b).total_seconds()) <= SAME_TIME_TOLERANCE_SEC


def decide(events: list[CalendarEvent], booking: BookingEvent) -> Decision:
    """Classify an incoming booking against current calendar state."""
    match = match_booking(events, booking)

    if match.is_ambiguous:
        return Decision(
            action=ActionType.NOOP,
            reason=Reason.AMBIGUOUS_MATCH,
            booking=booking,
            candidates=tuple(e.event_id for e in match.candidates),
            note="More than one calendar event could be this booking; asking the user rather than guessing.",
        )

    if booking.status is BookingStatus.CANCELLED:
        if match.matched is None:
            # NFR2: a cancellation for an event we already removed is a no-op,
            # not an error and not a second confirmation prompt.
            return Decision(
                action=ActionType.NOOP,
                reason=Reason.ALREADY_CANCELLED,
                booking=booking,
                note="No calendar event matches this cancellation; nothing to remove.",
            )
        return Decision(
            action=ActionType.DELETE,
            reason=Reason.CANCELLATION,
            booking=booking,
            target_event_id=match.matched.event_id,
            payload=match.matched,
        )

    if match.matched is None:
        return Decision(
            action=ActionType.CREATE,
            reason=Reason.NEW_BOOKING,
            booking=booking,
            payload=_event_from(booking),
        )

    existing = match.matched
    if _same_time(existing.start_time, booking.date_time):
        # Idempotency (NFR2): the same notification arriving twice changes nothing.
        return Decision(
            action=ActionType.NOOP,
            reason=Reason.DUPLICATE_ALREADY_APPLIED,
            booking=booking,
            target_event_id=existing.event_id,
            note="Calendar already reflects this booking.",
        )

    return Decision(
        action=ActionType.UPDATE,
        reason=Reason.RESCHEDULE,
        booking=booking,
        target_event_id=existing.event_id,
        payload=_merge_for_update(existing, booking),
    )


def _merge_for_update(existing: CalendarEvent, booking: BookingEvent) -> CalendarEvent:
    """A reschedule moves an event; it does not rename it.

    The user recognises the entry by its title, and a change notice often carries
    less detail than the original confirmation did -- so taking the incoming title
    wholesale silently downgrades "Flight AI440 Bengaluru (BLR) to Chennai (MAA)"
    to "Flight AI440-XY12". Time and any newly-supplied detail move; the title,
    and anything the notice omits, stay.
    """
    updated = _event_from(booking, event_id=existing.event_id)
    return CalendarEvent(
        event_id=existing.event_id,
        title=existing.title or updated.title,
        start_time=updated.start_time,
        end_time=updated.end_time or existing.end_time,
        booking_ref=updated.booking_ref or existing.booking_ref,
        booking_type=updated.booking_type or existing.booking_type,
        location=updated.location or existing.location,
        alarm=updated.alarm,
    )


def _event_from(booking: BookingEvent, event_id: str | None = None) -> CalendarEvent:
    ev = CalendarEvent(
        event_id=event_id or "",
        title=booking.title,
        start_time=booking.date_time,
        end_time=booking.end_time,
        booking_ref=booking.booking_ref,
        booking_type=booking.booking_type,
        location=booking.location,
    )
    if booking.alarm_offset_min:
        ev = ev.with_alarm_offset(booking.alarm_offset_min, alarm_id=f"AL-{booking.booking_id}")
    return ev


@dataclass
class ReconciliationEngine:
    """Applies decisions. The destructive path is unreachable without consent."""

    calendar: CalendarProvider

    def apply(self, decision: Decision) -> CalendarEvent | None:
        """Non-destructive execution only. Raises on a destructive decision."""
        if decision.is_destructive:
            raise PermissionError(
                f"{decision.action.value} is destructive and requires apply_destructive() "
                f"with a ConfirmedDecision (FR5)."
            )
        if decision.action is ActionType.CREATE and decision.payload is not None:
            return self.calendar.create_event(decision.payload)
        return None

    def apply_destructive(self, confirmed: ConfirmedDecision) -> CalendarEvent | None:
        """Executes DELETE/UPDATE. The signature is the FR5 gate.

        A ConfirmedDecision cannot be constructed anywhere except
        ConfirmationRequest.confirm(), so no caller can reach this with an
        unapproved decision — including a future caller who forgets the rule.
        """
        decision = confirmed.decision
        if decision.action is ActionType.DELETE and decision.target_event_id:
            self.calendar.delete_event(decision.target_event_id)
            return None
        if decision.action is ActionType.UPDATE and decision.payload is not None:
            return self.calendar.update_event(decision.payload)
        return None
