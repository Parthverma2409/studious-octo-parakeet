"""CalendarProvider port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from packages.core.models import CalendarEvent


class CalendarError(RuntimeError):
    """Base for provider failures."""


class RateLimitError(CalendarError):
    """UC1 E3 — the write should be queued and the user told about the delay."""


@runtime_checkable
class CalendarProvider(Protocol):
    def list_events(self) -> list[CalendarEvent]: ...
    def create_event(self, event: CalendarEvent) -> CalendarEvent: ...
    def update_event(self, event: CalendarEvent) -> CalendarEvent: ...
    def delete_event(self, event_id: str) -> None: ...
