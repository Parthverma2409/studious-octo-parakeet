"""Extract structured booking data from a booking email (FR2, UC2).

Rule-based, like `baseline.py`, and for the same two reasons: it is the floor an
LLM extractor must beat, and it is the backstop when one returns unusable output.

The first job is not extraction at all -- it is deciding whether the email is a
booking notification in the first place. UC2's stated open issue is telling
marketing mail from genuine change mail, and a false positive here writes into a
real calendar. So classification is conservative: anything that smells
promotional is dropped before a single field is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from packages.core.models import BookingStatus, BookingType

from .resolve import resolve_datetime

#: Marketing tells on itself. Checked before anything else, and a hit means the
#: email is discarded outright rather than parsed and then second-guessed.
_PROMO = re.compile(
    r"(unsubscribe|save \d+%|off your next|limited[- ]time|book now and|fare sale|"
    r"deals? of the (day|week)|earn \d+ points|exclusive offer|don'?t miss out)",
    re.I,
)

_STATUS_RULES: list[tuple[BookingStatus, re.Pattern]] = [
    (BookingStatus.CANCELLED, re.compile(r"\b(cancelled|canceled|cancellation confirmed|has been cancelled)\b", re.I)),
    (BookingStatus.RESCHEDULED, re.compile(r"\b(rescheduled|reschedule|time change|schedule change|new departure|has been moved|revised timing)\b", re.I)),
    (BookingStatus.CONFIRMED, re.compile(r"\b(confirmed|confirmation|booking is confirmed|e-?ticket|itinerary)\b", re.I)),
]

_TYPE_RULES: list[tuple[BookingType, re.Pattern]] = [
    (BookingType.FLIGHT, re.compile(r"\b(flight|airline|departure|boarding|pnr)\b", re.I)),
    (BookingType.TRAIN, re.compile(r"\b(train|rail|irctc|coach)\b", re.I)),
    (BookingType.HOTEL, re.compile(r"\b(hotel|check-?in|room|nights?)\b", re.I)),
    (BookingType.APPOINTMENT, re.compile(r"\b(appointment|clinic|consultation|doctor|dr\.?)\b", re.I)),
]

# The label is matched case-insensitively; the reference itself is NOT. Under a
# blanket re.I the bare-6-character alternative happily matched the word "number"
# out of "Confirmation number:", and a wrong booking_ref is worse than none --
# it produces a false exact match, which is how the wrong event gets deleted.
# The bare form therefore also has to contain a digit, as real references do.
_REF = re.compile(
    r"(?i:\b(?:booking(?:\s+ref(?:erence)?)?|ref|pnr|confirmation)\b\s*(?:#|no\.?|number)?\s*:?\s*)"
    r"([A-Z0-9]{2,8}-[A-Z0-9]{2,8}|(?=[A-Z0-9]*\d)[A-Z0-9]{6})\b"
)
_ISO_DT = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{1,2}:\d{2})\b")
_DATE_AT_TIME = re.compile(r"\b(\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?)\s+(?:at\s+)?(\d{1,2}[:.]\d{2}\s*(?:am|pm)?)\b", re.I)
_TITLE = re.compile(r"^\s*(?:subject:\s*)?(.+?)\s*$", re.I)
# The vendor's own itinerary line, e.g. "Flight: AI440 Bengaluru (BLR) to Chennai (MAA)".
_ITINERARY = re.compile(r"^\s*(flight|train|hotel|appointment)\s*:\s*(.+)$", re.I | re.M)


@dataclass
class EmailExtraction:
    """What UC2 step 2 produces. `is_booking_related=False` stops the pipeline."""

    is_booking_related: bool
    status: BookingStatus | None = None
    booking_type: BookingType | None = None
    booking_ref: str | None = None
    date_time: datetime | None = None
    title: str | None = None
    reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        """Enough to build a BookingEvent the reconciler can use."""
        return bool(self.is_booking_related and self.status and self.date_time and self.title)


def extract_email(subject: str, body: str, now: datetime) -> EmailExtraction:
    blob = "%s\n%s" % (subject, body)

    if _PROMO.search(blob):
        # UC2 open issue. Dropping a genuine change email is recoverable -- the
        # vendor resends, and the user can still act. Acting on a marketing email
        # is not: it raises a destructive prompt about a booking nothing changed.
        return EmailExtraction(False, reason="promotional", notes=["matched a marketing pattern"])

    status = next((s for s, pattern in _STATUS_RULES if pattern.search(blob)), None)
    if status is None:
        return EmailExtraction(False, reason="no booking status found")

    booking_type = next((t for t, pattern in _TYPE_RULES if pattern.search(blob)), None)

    ref_match = _REF.search(blob)
    booking_ref = ref_match.group(1).upper() if ref_match else None

    date_time, notes = _find_datetime(blob, now)

    title = _build_title(blob, booking_type, booking_ref, subject)

    extraction = EmailExtraction(
        is_booking_related=True,
        status=status,
        booking_type=booking_type,
        booking_ref=booking_ref,
        date_time=date_time,
        title=title or None,
        notes=notes,
    )
    if not extraction.is_actionable:
        # UC2 E1: flagged for manual review, calendar untouched.
        missing = [
            name
            for name, value in (("status", status), ("date_time", date_time), ("title", title))
            if not value
        ]
        extraction.reason = "unparseable: missing %s" % ", ".join(missing)
    return extraction


def _build_title(blob: str, booking_type: BookingType | None, booking_ref: str | None, subject: str) -> str | None:
    """Name the BOOKING, not the email.

    The subject line describes the notification -- "Schedule change for your
    flight AI440" -- and putting that in a calendar leaves the user with an event
    named after a piece of admin. Prefer the itinerary line the vendor already
    provides, then a type-and-reference label, and only fall back to the subject.
    """
    match = _ITINERARY.search(blob)
    if match:
        detail = match.group(2).strip(" .")
        label = match.group(1).capitalize()
        return detail if detail.lower().startswith(label.lower()) else "%s %s" % (label, detail)

    if booking_type and booking_ref:
        return "%s %s" % (booking_type.value.capitalize(), booking_ref)
    if booking_type:
        return booking_type.value.capitalize() + " booking"

    stripped = _TITLE.match(subject)
    return (stripped.group(1) if stripped else subject.strip()) or None


def _find_datetime(blob: str, now: datetime) -> tuple[datetime | None, list[str]]:
    match = _ISO_DT.search(blob)
    if match:
        resolution = resolve_datetime(match.group(1), match.group(2), now)
        if resolution.ok:
            return resolution.value, []

    match = _DATE_AT_TIME.search(blob)
    if match:
        date_text = re.sub(r"\s+\d{4}$", "", match.group(1))  # resolver infers the year
        resolution = resolve_datetime(date_text, match.group(2), now)
        if resolution.ok:
            return resolution.value, [resolution.note] if resolution.note else []

    return None, ["no resolvable date/time in the email"]
