"""Date/time resolution.

The governing rule under test: an ambiguous value resolves to None, never to a
plausible guess. Every "returns None" test here is a booking the agent will ask
about instead of silently getting wrong.
"""

from __future__ import annotations

from datetime import date, datetime, time

import pytest

from packages.nlp.resolve import resolve_date, resolve_datetime, resolve_time

# A Tuesday.
NOW = datetime(2026, 8, 18, 8, 0)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("9:15 am", time(9, 15)),
        ("6:40 pm", time(18, 40)),
        ("18:40", time(18, 40)),
        ("12:00 am", time(0, 0)),
        ("12:30 pm", time(12, 30)),
        ("noon", time(12, 0)),
        ("midnight", time(0, 0)),
        ("half past eight", time(8, 30)),
        ("quarter to five", time(4, 45)),
        ("quarter past nine", time(9, 15)),
        ("early morning", time(7, 0)),
        ("late evening", time(21, 0)),
        ("7 in the morning", time(7, 0)),
        ("7 in the evening", time(19, 0)),
    ],
)
def test_times_resolve(text: str, expected: time) -> None:
    assert resolve_time(text) == expected


@pytest.mark.parametrize("text", ["at 7", "7", "sometime", "half past", "25:00", "9:75", "13 pm", ""])
def test_ambiguous_times_resolve_to_none(text: str) -> None:
    """"At 7" is 07:00 or 19:00 with nothing to choose between them."""
    assert resolve_time(text) is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("tomorrow", date(2026, 8, 19)),
        ("day after tomorrow", date(2026, 8, 20)),
        ("today", date(2026, 8, 18)),
        ("20 August", date(2026, 8, 20)),
        ("3rd September", date(2026, 9, 3)),
        ("August 20", date(2026, 8, 20)),
        ("2026-08-20", date(2026, 8, 20)),
        ("the 21st", date(2026, 8, 21)),
        ("Friday", date(2026, 8, 21)),
        ("next Friday", date(2026, 8, 28)),
        ("this Sunday", date(2026, 8, 23)),
    ],
)
def test_dates_resolve(text: str, expected: date) -> None:
    assert resolve_date(text, NOW) == expected


def test_a_weekday_named_on_that_weekday_means_next_week() -> None:
    """Said on a Tuesday, "Tuesday" means the coming one, not today."""
    assert resolve_date("Tuesday", NOW) == date(2026, 8, 25)


def test_a_past_day_and_month_rolls_to_next_year() -> None:
    """"3 January" said in August is next January, not eight months ago."""
    assert resolve_date("3 January", NOW) == date(2027, 1, 3)


def test_an_ordinal_already_past_this_month_rolls_forward() -> None:
    assert resolve_date("the 5th", NOW) == date(2026, 9, 5)


@pytest.mark.parametrize("text", ["soon", "later", "next", "the 32nd", "45 August", "", None])
def test_unresolvable_dates_return_none(text) -> None:
    assert resolve_date(text, NOW) is None


def test_a_missing_date_is_never_defaulted() -> None:
    """The field that decides which day the user turns up is never invented."""
    resolution = resolve_datetime(None, "9:15 am", NOW)
    assert not resolution.ok
    assert resolution.value is None


def test_a_missing_time_is_defaulted_and_the_substitution_is_reported() -> None:
    resolution = resolve_datetime("20 August", None, NOW)
    assert resolution.ok
    assert resolution.value == datetime(2026, 8, 20, 9, 0)
    assert "assumed" in resolution.note


def test_an_unresolvable_time_is_reported_not_hidden() -> None:
    resolution = resolve_datetime("20 August", "at 7", NOW)
    assert resolution.ok
    assert "could not resolve time" in resolution.note


def test_date_and_time_combine() -> None:
    assert resolve_datetime("20 August", "6:40 pm", NOW).value == datetime(2026, 8, 20, 18, 40)
