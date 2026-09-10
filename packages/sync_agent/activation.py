"""ActivationWindow -- the hours the assistant is allowed to interrupt (NFR4).

NFR4 says the traveller configures an off-work window and the assistant works
within it. Two consequences, and the second is the one that carries risk:

  1. A command arriving outside the window is refused. Nothing is changed, and
     the user is told when the assistant is next awake.
  2. A destructive change that needs confirmation while the user is unreachable
     must NOT be applied, and must NOT be dropped. UC3 E2 is explicit: it is
     queued for the next active window. Losing it means a cancelled flight stays
     on the calendar; auto-applying it means an event is deleted without consent,
     which is an FR5 violation. Queuing is the only correct third option.

The midnight-crossing case is the whole difficulty here. "20:00 to 07:00" is the
normal shape of an off-work window, and the naive ``start <= t <= end`` test says
that window is empty -- so every command would be refused and every confirmation
queued forever. It is handled explicitly and tested explicitly.

This module holds no clock: every entry point takes ``now``, matching the rest of
the codebase, so the behaviour is replayable in the scenario suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from packages.core.confirmation import ConfirmationRequest, ConfirmationStatus, ConfirmationStore


@dataclass(frozen=True)
class ActivationWindow:
    """The daily period during which the assistant may act and speak.

    ``start == end`` means always active. That is the conventional reading of a
    00:00-00:00 window and it gives a configuration file a way to say "no
    restriction" without a second flag to keep in sync.
    """

    start: time
    end: time

    @classmethod
    def parse(cls, spec: str) -> "ActivationWindow":
        """Build from a ``"HH:MM-HH:MM"`` string, as a config file would hold it."""
        try:
            raw_start, raw_end = spec.split("-")
            return cls(time.fromisoformat(raw_start.strip()), time.fromisoformat(raw_end.strip()))
        except ValueError as exc:
            raise ValueError("activation window must look like '20:00-07:00', got %r" % spec) from exc

    @property
    def crosses_midnight(self) -> bool:
        return self.end < self.start

    @property
    def always_active(self) -> bool:
        return self.start == self.end

    def contains(self, when: datetime) -> bool:
        """Is ``when`` inside the window?

        Half-open on purpose: the start instant is inside, the end instant is
        outside. A closed interval would make 07:00 belong to both a 20:00-07:00
        window and the 07:00-09:00 one after it, and "which window am I in" is
        not a question that should have two answers.
        """
        if self.always_active:
            return True
        moment = when.time()
        if self.crosses_midnight:
            # Two disjoint pieces of the clock face: [start, midnight) and
            # [midnight, end). This is the branch that is easy to omit.
            return moment >= self.start or moment < self.end
        return self.start <= moment < self.end

    def next_start(self, when: datetime) -> datetime:
        """The next instant the window opens, at or after ``when``.

        Used as the deadline anchor for a queued confirmation, so it must never
        return a time in the past: a queued request whose window opened yesterday
        would expire the moment it was released.
        """
        if self.always_active:
            return when
        today = when.date()
        candidate = datetime.combine(today, self.start)
        if candidate < when:
            candidate = datetime.combine(today + timedelta(days=1), self.start)
        return candidate

    def describe(self) -> str:
        if self.always_active:
            return "always"
        return "%s-%s" % (self.start.strftime("%H:%M"), self.end.strftime("%H:%M"))


def queue_outside_window(
    request: ConfirmationRequest, window: ActivationWindow | None, now: datetime
) -> bool:
    """UC3 E2. Park a confirmation the user cannot answer yet. True if parked.

    ``ConfirmationRequest.queue_for_active_window`` already exists in
    ``packages/core/confirmation.py`` and does the state change; this is the
    caller that was missing. The timeout is re-anchored to the next window start
    by that method, so a request raised at 02:00 against an 09:00-18:00 window
    gets its full response window from 09:00 rather than expiring while the user
    sleeps.
    """
    if window is None or window.contains(now):
        return False
    request.queue_for_active_window(window.next_start(now))
    return True


def find_open_confirmation(
    store: ConfirmationStore, booking_ref: str | None, target_event_id: str | None, now: datetime
) -> ConfirmationRequest | None:
    """NFR2, extended across the window boundary.

    ``ConfirmationStore.find_pending_for`` only sees PENDING requests, so once a
    confirmation is queued a repeat of the same notification would open a
    *second* one -- the exact duplicate-prompt bug NFR2 exists to prevent, moved
    one status along. Queued requests are checked here on the same terms.
    """
    existing = store.find_pending_for(booking_ref, target_event_id, now)
    if existing is not None:
        return existing
    for request in store.all():
        if request.status is not ConfirmationStatus.QUEUED:
            continue
        booking = request.decision.booking
        same_ref = booking_ref is not None and booking is not None and booking.booking_ref == booking_ref
        same_target = target_event_id is not None and request.decision.target_event_id == target_event_id
        if same_ref or same_target:
            return request
    return None


def release_queued(store: ConfirmationStore, window: ActivationWindow | None, now: datetime) -> list[ConfirmationRequest]:
    """Move QUEUED requests back to PENDING once the window is open again.

    Without this, a queued request is invisible to ``ConfirmationStore.pending``
    forever and a later "yes" finds nothing to apply -- the change would be
    parked, not delivered, which is only half of UC3 E2.

    The status is flipped here rather than by adding a ``release`` method to
    ``confirmation.py``: that module is the frozen FR5 contract, and the
    scheduling policy that decides *when* a user is reachable belongs to the
    activation layer, not to the safety gate. Timeouts are untouched --
    ``queue_for_active_window`` already re-anchored them, and a request whose
    re-anchored deadline has also passed will be expired by the next
    ``pending(now)`` call, which is the UC3 E1 fail-safe doing its job.
    """
    if window is None or not window.contains(now):
        return []
    released = []
    for request in store.all():
        if request.status is ConfirmationStatus.QUEUED:
            request.status = ConfirmationStatus.PENDING
            released.append(request)
    return released
