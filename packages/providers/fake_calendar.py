"""In-memory CalendarProvider: deterministic, inspectable, free.

Also records a mutation log, which is what the scenario runner asserts against
for NFR2 (a duplicate notification must produce exactly one mutation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from packages.core.models import CalendarEvent

from .calendar import RateLimitError


@dataclass
class FakeCalendar:
    events: dict[str, CalendarEvent] = field(default_factory=dict)
    mutations: list[tuple[str, str]] = field(default_factory=list)
    _seq: int = 0
    #: When > 0, the next N write calls raise RateLimitError (UC1 E3).
    fail_next_writes: int = 0

    def seed(self, event: CalendarEvent) -> CalendarEvent:
        """Load initial state without recording a mutation."""
        self.events[event.event_id] = event
        return event

    def list_events(self) -> list[CalendarEvent]:
        return sorted(self.events.values(), key=lambda e: (e.start_time, e.event_id))

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        self._maybe_fail()
        if not event.event_id:
            self._seq += 1
            event = CalendarEvent(**{**event.__dict__, "event_id": f"E{self._seq + 100}"})
        self.events[event.event_id] = event
        self.mutations.append(("CREATE", event.event_id))
        return event

    def update_event(self, event: CalendarEvent) -> CalendarEvent:
        self._maybe_fail()
        if event.event_id not in self.events:
            raise KeyError(event.event_id)
        self.events[event.event_id] = event
        self.mutations.append(("UPDATE", event.event_id))
        return event

    def delete_event(self, event_id: str) -> None:
        self._maybe_fail()
        self.events.pop(event_id, None)
        self.mutations.append(("DELETE", event_id))

    def _maybe_fail(self) -> None:
        if self.fail_next_writes > 0:
            self.fail_next_writes -= 1
            raise RateLimitError("Calendar API rate limit exceeded")
