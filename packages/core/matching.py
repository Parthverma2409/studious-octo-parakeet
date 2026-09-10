"""Booking-to-calendar-event matching (FR3).

Three tiers, strongest first:
  1. vendor booking reference  -> exact, high confidence
  2. type + same day + fuzzy title/destination overlap -> heuristic
  3. nothing                   -> no match

Tier 2 can return several candidates. Ambiguity is reported, never guessed:
picking one of two plausible events and then deleting it is precisely the
failure FR5 exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import timedelta

from .models import BookingEvent, CalendarEvent

#: Two events of the same type this close together are treated as the same booking
#: moved in time, rather than two separate bookings.
RESCHEDULE_WINDOW = timedelta(hours=36)

_STOPWORDS = {"flight", "train", "hotel", "appointment", "to", "from", "the", "at", "with", "my"}


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS}


def _overlap(a: str | None, b: str | None) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_by_phrase(events: list[CalendarEvent], phrase: str, *, threshold: float = 0.25) -> list[CalendarEvent]:
    """Events a spoken reference like "my Chennai flight" could mean.

    Used before a BookingEvent exists: "cancel my Chennai flight" carries no date,
    so the calendar has to supply it. Returns every plausible candidate rather
    than a best guess -- the caller asks the user when there is more than one.
    """
    scored = [(_overlap(ev.title, phrase), ev) for ev in events]
    hits = [(score, ev) for score, ev in scored if score >= threshold]
    if not hits:
        return []
    best = max(score for score, _ in hits)
    return [ev for score, ev in hits if abs(score - best) < 1e-9]


@dataclass(frozen=True)
class MatchResult:
    matched: CalendarEvent | None
    candidates: tuple[CalendarEvent, ...] = ()
    confidence: float = 0.0
    by_reference: bool = False

    @property
    def is_ambiguous(self) -> bool:
        return self.matched is None and len(self.candidates) > 1


def match_booking(events: list[CalendarEvent], booking: BookingEvent, *, title_threshold: float = 0.34) -> MatchResult:
    """Find the calendar event this booking refers to, if any."""
    if booking.booking_ref:
        exact = [e for e in events if e.booking_ref and e.booking_ref == booking.booking_ref]
        if exact:
            return MatchResult(matched=exact[0], candidates=tuple(exact), confidence=1.0, by_reference=True)

    scored: list[tuple[float, CalendarEvent]] = []
    for ev in events:
        if booking.booking_type and ev.booking_type and ev.booking_type != booking.booking_type:
            continue
        if abs(ev.start_time - booking.date_time) > RESCHEDULE_WINDOW:
            continue
        score = max(
            _overlap(ev.title, booking.title),
            _overlap(ev.title, booking.destination),
            _overlap(ev.location, booking.location),
        )
        if ev.start_time == booking.date_time:
            score = max(score, 0.5)
        if score >= title_threshold:
            scored.append((score, ev))

    if not scored:
        return MatchResult(matched=None)

    scored.sort(key=lambda p: (-p[0], p[1].start_time))
    best_score, best_event = scored[0]
    tied = [ev for score, ev in scored if abs(score - best_score) < 1e-9]

    if len(tied) > 1:
        # Genuinely ambiguous — hand the candidates up rather than choosing.
        return MatchResult(matched=None, candidates=tuple(tied), confidence=best_score)

    return MatchResult(matched=best_event, candidates=tuple(ev for _, ev in scored), confidence=best_score)
