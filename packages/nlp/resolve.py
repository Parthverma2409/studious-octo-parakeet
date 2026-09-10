"""Turn extracted slot text into a concrete datetime.

The governing rule: **when a value cannot be resolved confidently, return None.**
A missing date makes the agent ask a clarifying question (UC1 E2). A guessed date
creates a calendar entry the user will trust and act on. Under-resolving is
recoverable; a confident wrong answer is not.

Deliberately not `dateparser`: it resolves almost anything to something, which is
the opposite of what this layer needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
#: Vague parts of day. These are conventions, not measurements -- the agent says
#: which one it used so the user can correct it.
_DAY_PARTS = {
    "early morning": time(7, 0),
    "morning": time(9, 0),
    "noon": time(12, 0),
    "afternoon": time(14, 0),
    "late evening": time(21, 0),
    "evening": time(18, 0),
    "night": time(21, 0),
    "midnight": time(0, 0),
}

_DAY_ORDINAL = re.compile(r"^(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?$", re.I)
_DAY_MONTH = re.compile(r"^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)$", re.I)
_MONTH_DAY = re.compile(r"^([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?$", re.I)
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")

_CLOCK = re.compile(r"^(\d{1,2})[:.](\d{2})\s*(am|pm)?$", re.I)
_HOUR_MERIDIEM = re.compile(r"^(\d{1,2})\s*(am|pm)$", re.I)
_HOUR_IN_PART = re.compile(r"^(\d{1,2})\s+in\s+the\s+(morning|afternoon|evening|night)$", re.I)
_HALF_PAST = re.compile(r"^half\s+past\s+(\w+)$", re.I)
_QUARTER = re.compile(r"^quarter\s+(past|to)\s+(\w+)$", re.I)


@dataclass(frozen=True)
class Resolution:
    """A resolved datetime, or an explanation of why it could not be resolved."""

    value: datetime | None
    note: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None


def _as_hour(token: str) -> int | None:
    token = token.strip().lower()
    if token.isdigit():
        hour = int(token)
        return hour if 0 <= hour <= 23 else None
    return _NUMBER_WORDS.get(token)


def resolve_time(text: str | None) -> time | None:
    """Resolve a time expression. Returns None when genuinely ambiguous."""
    if not text:
        return None
    value = text.strip().lower()

    if value in _DAY_PARTS:
        return _DAY_PARTS[value]

    match = _CLOCK.match(value)
    if match:
        hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
        if meridiem:
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if meridiem.lower() == "pm" else 0)
        elif not 0 <= hour <= 23:
            return None
        return time(hour, minute) if minute < 60 else None

    match = _HOUR_MERIDIEM.match(value)
    if match:
        hour = int(match.group(1))
        if not 1 <= hour <= 12:
            return None
        return time(hour % 12 + (12 if match.group(2).lower() == "pm" else 0), 0)

    match = _HOUR_IN_PART.match(value)
    if match:
        hour, part = int(match.group(1)), match.group(2).lower()
        if not 1 <= hour <= 12:
            return None
        return time(hour % 12 + (0 if part == "morning" else 12), 0)

    match = _HALF_PAST.match(value)
    if match:
        hour = _as_hour(match.group(1))
        return time(hour, 30) if hour is not None and hour < 24 else None

    match = _QUARTER.match(value)
    if match:
        direction, hour = match.group(1).lower(), _as_hour(match.group(2))
        if hour is None:
            return None
        if direction == "past":
            return time(hour, 15) if hour < 24 else None
        return time((hour - 1) % 24, 45)

    # A bare "at 7" is 07:00 or 19:00 with no way to tell. Refuse it.
    return None


def resolve_date(text: str | None, now: datetime) -> date | None:
    """Resolve a date expression relative to `now`. Returns None when ambiguous."""
    if not text:
        return None
    value = " ".join(text.strip().lower().split())
    today = now.date()

    if value == "today":
        return today
    if value == "tomorrow":
        return today + timedelta(days=1)
    if value in ("day after tomorrow", "the day after tomorrow"):
        return today + timedelta(days=2)

    match = _ISO_DATE.match(value)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None

    if value in ("this weekend", "next weekend"):
        saturday = today + timedelta(days=(5 - today.weekday()) % 7 or 7)
        return saturday + timedelta(days=7) if value == "next weekend" else saturday

    # "Monday" / "this Monday" / "next Monday". "next" always means the week
    # after the coming one, which is what people mean and what a naive
    # implementation gets wrong when today is already that weekday.
    weekday_match = re.match(r"^(this|next|coming)?\s*([a-z]+)$", value)
    if weekday_match and weekday_match.group(2) in _WEEKDAYS:
        qualifier, name = weekday_match.group(1), weekday_match.group(2)
        ahead = (_WEEKDAYS[name] - today.weekday()) % 7
        if ahead == 0:
            ahead = 7  # "Monday" said on a Monday means the next one
        target = today + timedelta(days=ahead)
        return target + timedelta(days=7) if qualifier == "next" else target

    match = _DAY_MONTH.match(value)
    if match and match.group(2) in _MONTHS:
        return _with_year(int(match.group(1)), _MONTHS[match.group(2)], today)

    match = _MONTH_DAY.match(value)
    if match and match.group(1) in _MONTHS:
        return _with_year(int(match.group(2)), _MONTHS[match.group(1)], today)

    match = _DAY_ORDINAL.match(value)
    if match:
        day = int(match.group(1))
        for month_offset in (0, 1):
            month = today.month + month_offset
            year = today.year + (month - 1) // 12
            month = (month - 1) % 12 + 1
            try:
                candidate = date(year, month, day)
            except ValueError:
                continue
            if candidate >= today:
                return candidate
        return None

    return None


def _with_year(day: int, month: int, today: date) -> date | None:
    """Bare day+month means the next occurrence -- bookings are in the future."""
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate >= today:
            return candidate
    return None


def resolve_datetime(
    date_text: str | None,
    time_text: str | None,
    now: datetime,
    default_time: time = time(9, 0),
) -> Resolution:
    """Combine a date and time expression into one datetime.

    A missing time is filled with a stated default and the substitution is
    reported, so the agent can say "on 20 August at 9am" rather than silently
    inventing an hour. A missing DATE is never defaulted -- that is the field
    that decides which day a user turns up.
    """
    resolved_date = resolve_date(date_text, now)
    if resolved_date is None:
        return Resolution(None, "could not resolve a date from %r" % date_text if date_text else "no date given")

    resolved_time = resolve_time(time_text)
    note = None
    if resolved_time is None:
        resolved_time = default_time
        note = (
            "no time given, assumed %s" % default_time.strftime("%H:%M")
            if not time_text
            else "could not resolve time %r, assumed %s" % (time_text, default_time.strftime("%H:%M"))
        )

    return Resolution(datetime.combine(resolved_date, resolved_time), note)
