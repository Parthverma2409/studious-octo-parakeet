"""Reconciliation decisions and the FR5 destructive-action gate.

The gate is structural, not a policy check. ``apply_destructive`` accepts only a
``ConfirmedDecision``, and the sole way to construct one is
``ConfirmationRequest.resolve(confirmed=True)``. Forgetting to check a flag is a
possible bug; there is no code path here that can delete an event without a
resolved confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .models import BookingEvent, CalendarEvent


class ActionType(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


#: Actions that destroy or materially alter user data. FR5 applies to these.
DESTRUCTIVE = frozenset({ActionType.DELETE, ActionType.UPDATE})


class Reason(str, Enum):
    """Why the engine decided what it did — surfaced to the user and the evaluator."""

    NEW_BOOKING = "new_booking"
    DUPLICATE_ALREADY_APPLIED = "duplicate_already_applied"
    RESCHEDULE = "reschedule"
    CANCELLATION = "cancellation"
    AMBIGUOUS_MATCH = "ambiguous_match"
    UNPARSEABLE = "unparseable"
    ALREADY_CANCELLED = "already_cancelled"


@dataclass(frozen=True)
class Decision:
    """What should happen to the calendar. Produced by a pure function."""

    action: ActionType
    reason: Reason
    booking: BookingEvent | None = None
    target_event_id: str | None = None
    payload: CalendarEvent | None = None
    #: Candidate events when the match was ambiguous (FR3 / UC1 open issue).
    candidates: tuple[str, ...] = ()
    note: str | None = None

    @property
    def is_destructive(self) -> bool:
        return self.action in DESTRUCTIVE

    @property
    def requires_confirmation(self) -> bool:
        """FR5: destructive changes, and only those, need explicit user consent."""
        return self.is_destructive

    def describe(self) -> str:
        """Unambiguous wording for the confirmation prompt (SRS UI priority 1)."""
        if self.action is ActionType.DELETE:
            return f"Delete the calendar event '{self._target_title()}' and cancel its alarm?"
        if self.action is ActionType.UPDATE:
            when = self.payload.start_time.strftime("%d %b %Y at %H:%M") if self.payload else "a new time"
            return f"Move the calendar event '{self._target_title()}' to {when}?"
        return f"{self.action.value.lower()} {self._target_title()}"

    def _target_title(self) -> str:
        if self.payload is not None:
            return self.payload.title
        if self.booking is not None:
            return self.booking.title
        return self.target_event_id or "the event"


class _Token:
    """Unforgeable capability. Held only by the confirmation module."""

    __slots__ = ()


_GRANT = _Token()


@dataclass(frozen=True)
class ConfirmedDecision:
    """A Decision the user has explicitly approved.

    Constructing this outside ``ConfirmationRequest.resolve`` raises. It is the
    only key that opens ``apply_destructive``.
    """

    decision: Decision
    confirmation_id: str
    confirmed_at: datetime
    _token: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _GRANT:
            raise PermissionError(
                "ConfirmedDecision may only be created by ConfirmationRequest.resolve() "
                "(FR5: no destructive action without explicit user confirmation)."
            )


def _grant(decision: Decision, confirmation_id: str, when: datetime) -> ConfirmedDecision:
    """Internal factory. Imported only by packages.core.confirmation."""
    return ConfirmedDecision(decision, confirmation_id, when, _GRANT)
