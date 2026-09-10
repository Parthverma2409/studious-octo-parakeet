"""Email extraction (FR2, UC2).

The first decision -- is this a booking notification at all? -- matters more than
any field it extracts. A false positive on marketing mail raises a destructive
prompt about a booking that never changed.
"""

from __future__ import annotations

from datetime import datetime
from email import message_from_string
from pathlib import Path

import pytest

from packages.core.models import BookingStatus, BookingType
from packages.nlp.email_extract import extract_email

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "emails" / "fixtures"
NOW = datetime(2026, 8, 18, 8, 0)


def _fixture(name: str):
    msg = message_from_string((FIXTURES / name).read_text(encoding="utf-8"))
    return msg.get("Subject", ""), str(msg.get_payload())


def test_cancellation_is_read_correctly() -> None:
    subject, body = _fixture("cancel_ai440.eml")
    result = extract_email(subject, body, NOW)
    assert result.is_actionable
    assert result.status is BookingStatus.CANCELLED
    assert result.booking_type is BookingType.FLIGHT
    assert result.booking_ref == "AI440-XY12"
    assert result.date_time == datetime(2026, 8, 20, 9, 15)


def test_confirmation_is_read_correctly() -> None:
    subject, body = _fixture("confirm_ai440.eml")
    result = extract_email(subject, body, NOW)
    assert result.is_actionable
    assert result.status is BookingStatus.CONFIRMED
    assert result.booking_ref == "AI440-XY12"


def test_reschedule_picks_up_the_new_time() -> None:
    subject, body = _fixture("reschedule_ai440.eml")
    result = extract_email(subject, body, NOW)
    assert result.status is BookingStatus.RESCHEDULED
    assert result.date_time == datetime(2026, 8, 20, 14, 45)


def test_promotional_mail_is_rejected_despite_looking_like_a_cancellation() -> None:
    """Subject: "Your booking is cancelled? Rebook now and save 30%" -- with a
    real-looking departure line in the body. UC2's stated open issue."""
    subject, body = _fixture("promo_fare_sale.eml")
    result = extract_email(subject, body, NOW)
    assert not result.is_booking_related
    assert result.reason == "promotional"


def test_email_without_a_date_is_unparseable_not_guessed() -> None:
    subject, body = _fixture("unparseable_itinerary.eml")
    result = extract_email(subject, body, NOW)
    assert result.is_booking_related
    assert not result.is_actionable
    assert "date_time" in result.reason


def test_ordinary_mail_is_not_booking_related() -> None:
    result = extract_email("Lunch tomorrow?", "Are you free at 1pm?", NOW)
    assert not result.is_booking_related


def test_cancellation_beats_confirmation_when_both_words_appear() -> None:
    """"Your confirmed booking has been cancelled" is a cancellation. Reading it
    as a confirmation would leave a dead event in the calendar."""
    result = extract_email(
        "Update on your confirmed booking",
        "Your confirmed flight has been cancelled.\nBooking reference: AI440-XY12\nDeparture: 2026-08-20 09:15",
        NOW,
    )
    assert result.status is BookingStatus.CANCELLED


@pytest.mark.parametrize(
    "body,expected",
    [
        ("Booking reference: AI440-XY12\n", "AI440-XY12"),
        ("PNR: 6E211-QQ01\n", "6E211-QQ01"),
        ("Confirmation number: TAJ-99871\n", "TAJ-99871"),
    ],
)
def test_reference_formats(body: str, expected: str) -> None:
    result = extract_email("Booking cancelled", body + "Departure: 2026-08-20 09:15", NOW)
    assert result.booking_ref == expected


def test_day_month_date_format_is_understood() -> None:
    result = extract_email(
        "Booking confirmed",
        "Your flight is confirmed.\nBooking reference: AI440-XY12\n20 August 2026 at 09:15\n",
        NOW,
    )
    assert result.date_time == datetime(2026, 8, 20, 9, 15)
