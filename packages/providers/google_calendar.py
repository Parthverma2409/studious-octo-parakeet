"""GoogleCalendarProvider -- the real CalendarProvider behind the same port (FR4).

The whole point of the port is that this file is the only thing that changes when
the demo stops using ``FakeCalendar``. ``SyncAgent`` and ``ReconciliationEngine``
do not import it, do not know it exists, and are not edited to accommodate it.

Two things this adapter has to get right that the fake does not:

  * **Field mapping.** Google's event resource has no notion of a booking
    reference or a booking type. They go into ``extendedProperties.private``,
    which is the documented place for application state and is what makes FR3
    matching survive a round trip through Google. Lose them and every
    re-notification looks like a new booking.
  * **Time zones.** The domain model uses naive datetimes throughout. Google
    always answers with an offset. Naive and aware datetimes compare by raising
    TypeError, so mixing them would turn a reschedule into a crash. Everything
    that comes back is converted to the configured zone and flattened to naive;
    everything sent out carries an explicit ``timeZone``. Default is UTC because
    ``zoneinfo`` has no tz database on a bare Windows install and a missing tzdb
    at 3am is a worse failure than an explicit UTC default.

Not exercised against Google here -- there are no credentials on this machine.
Everything is driven through an injected ``GoogleApiClient``, and the tests feed
it recorded payloads shaped like real Calendar API responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from packages.core.models import Alarm, BookingType, CalendarEvent

from .oauth import GoogleApiClient, GoogleApiError

API_ROOT = "https://www.googleapis.com/calendar/v3"

#: Google's own maximum for a single events.list page. Asking for the maximum
#: means fewer round trips; pagination still has to work, and it is tested.
PAGE_SIZE = 250

#: Where booking_ref / booking_type live on the Google resource.
PRIVATE_REF = "bookingRef"
PRIVATE_TYPE = "bookingType"


@dataclass
class GoogleCalendarProvider:
    """Implements ``CalendarProvider`` against Calendar API v3."""

    client: GoogleApiClient
    calendar_id: str = "primary"
    #: IANA name sent to Google with every write. See the module docstring for
    #: why the default is UTC rather than the machine's local zone.
    time_zone: str = "UTC"

    # -- CalendarProvider -------------------------------------------------

    def list_events(self) -> list[CalendarEvent]:
        """All non-cancelled events, every page of them.

        ``singleEvents=true`` expands recurring series into instances: an agent
        that returned the master of a weekly series and then deleted it would
        remove every occurrence, which is not what "cancel my Tuesday flight"
        means.
        """
        items = self.client.paginate(
            "%s/calendars/%s/events" % (API_ROOT, self.calendar_id),
            params={"maxResults": PAGE_SIZE, "singleEvents": "true", "orderBy": "startTime"},
        )
        events = [self._to_event(item) for item in items if item.get("status") != "cancelled"]
        # Same ordering contract as FakeCalendar, so scenario expectations and
        # tests do not depend on which provider is installed.
        return sorted(events, key=lambda e: (e.start_time, e.event_id))

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        created = self.client.request(
            "POST",
            "%s/calendars/%s/events" % (API_ROOT, self.calendar_id),
            payload=self._to_resource(event),
        )
        return self._to_event(created)

    def update_event(self, event: CalendarEvent) -> CalendarEvent:
        if not event.event_id:
            raise ValueError("update_event requires an event_id")
        updated = self.client.request(
            "PUT",
            "%s/calendars/%s/events/%s" % (API_ROOT, self.calendar_id, event.event_id),
            payload=self._to_resource(event),
        )
        return self._to_event(updated)

    def delete_event(self, event_id: str) -> None:
        self.client.request(
            "DELETE", "%s/calendars/%s/events/%s" % (API_ROOT, self.calendar_id, event_id)
        )

    # -- mapping ----------------------------------------------------------

    def _to_resource(self, event: CalendarEvent) -> dict:
        resource: dict = {
            "summary": event.title,
            "start": self._to_google_time(event.start_time),
            # Google rejects an event without an end. An hour is the convention
            # the rest of the system already uses for a flight with no stated
            # arrival; inventing nothing is not an option the API allows.
            "end": self._to_google_time(event.end_time or event.start_time + timedelta(hours=1)),
        }
        if event.location:
            resource["location"] = event.location
        private = {}
        if event.booking_ref:
            private[PRIVATE_REF] = event.booking_ref
        if event.booking_type is not None:
            private[PRIVATE_TYPE] = event.booking_type.value
        if private:
            resource["extendedProperties"] = {"private": private}
        if event.alarm is not None and event.alarm.is_active:
            minutes = int((event.start_time - event.alarm.trigger_time).total_seconds() // 60)
            resource["reminders"] = {
                "useDefault": False,
                "overrides": [{"method": "popup", "minutes": max(minutes, 0)}],
            }
        else:
            # An explicit empty override list is how the API is told "no alarm".
            # Omitting the key would leave the calendar's default reminder in
            # place, so a cancelled alarm would quietly survive (FR4).
            resource["reminders"] = {"useDefault": False, "overrides": []}
        return resource

    def _to_event(self, resource: dict) -> CalendarEvent:
        private = (resource.get("extendedProperties") or {}).get("private") or {}
        start = self._from_google_time(resource.get("start") or {})
        end = self._from_google_time(resource.get("end") or {})
        if start is None:
            # Every real event has a start. Failing loudly beats inventing a time
            # or dropping the event -- a silently missing event reads as "not on
            # the calendar", which is how a duplicate booking gets created.
            raise GoogleApiError(
                "calendar event %r has no start time" % resource.get("id", "<unknown>")
            )
        return CalendarEvent(
            event_id=resource.get("id", ""),
            title=resource.get("summary") or "(no title)",
            start_time=start,
            end_time=end,
            booking_ref=private.get(PRIVATE_REF),
            booking_type=_booking_type(private.get(PRIVATE_TYPE)),
            location=resource.get("location"),
            alarm=_alarm_from(resource, start),
        )

    def _to_google_time(self, when: datetime) -> dict:
        return {"dateTime": _to_naive(when).strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": self.time_zone}

    def _from_google_time(self, node: dict) -> datetime | None:
        raw = node.get("dateTime")
        if raw:
            return _to_naive(datetime.fromisoformat(raw.replace("Z", "+00:00")))
        raw = node.get("date")
        if raw:
            # An all-day booking. Midnight is the only defensible reading, and it
            # is what the alarm offset is then measured from.
            return datetime.combine(date.fromisoformat(raw), datetime.min.time())
        return None


def _to_naive(when: datetime) -> datetime:
    """Flatten to naive UTC. Naive input is assumed to already be in-zone."""
    if when.tzinfo is None:
        return when
    return when.astimezone(timezone.utc).replace(tzinfo=None)


def _booking_type(raw: str | None) -> BookingType | None:
    try:
        return BookingType(raw) if raw else None
    except ValueError:
        # An unknown type written by an older build must not break a read: the
        # event still exists and the user still needs to see it.
        return None


def _alarm_from(resource: dict, start: datetime | None) -> Alarm | None:
    reminders = resource.get("reminders") or {}
    overrides = reminders.get("overrides") or []
    if not overrides or start is None:
        return None
    minutes = overrides[0].get("minutes")
    if minutes is None:
        return None
    return Alarm(
        alarm_id="AL-%s" % resource.get("id", ""),
        trigger_time=start - timedelta(minutes=int(minutes)),
    )
