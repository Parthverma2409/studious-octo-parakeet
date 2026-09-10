"""ActivationWindow (NFR4) and the queued-confirmation path (UC3 E2).

Two properties are protected here, and the second is the one with teeth:

  * a window that crosses midnight -- "20:00 to 07:00" is the normal shape of an
    off-work window, and the naive `start <= t <= end` test says it is empty, so
    a bug here refuses every command and queues every confirmation forever;
  * a destructive change raised while the user is unreachable is neither applied
    nor lost. Applying it is an FR5 violation. Losing it leaves a cancelled
    flight on the calendar. Queuing is the only correct third option, and
    `ConfirmationRequest.queue_for_active_window` is the existing API for it.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import pytest

from packages.core.confirmation import DEFAULT_TIMEOUT, ConfirmationStatus, ConfirmationStore
from packages.core.decisions import ActionType
from packages.core.models import BookingType, CalendarEvent
from packages.providers.fake_calendar import FakeCalendar
from packages.providers.fake_inbox import FakeInbox
from packages.sync_agent.activation import (
    ActivationWindow,
    find_open_confirmation,
    queue_outside_window,
    release_queued,
)
from packages.sync_agent.agent import SyncAgent
from packages.sync_agent.email_watcher import EmailIngestionModule

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "emails" / "fixtures"

#: A conventional office window, and the one that does not wrap.
OFFICE = ActivationWindow(time(9, 0), time(18, 0))
#: The awkward one: the traveller is only reachable outside work hours.
EVENING = ActivationWindow(time(20, 0), time(7, 0))


def _day(hour: int, minute: int = 0, day: int = 18) -> datetime:
    return datetime(2026, 8, day, hour, minute)


# -- the window itself ------------------------------------------------------


@pytest.mark.parametrize(
    "moment,expected",
    [
        (_day(8, 59), False),
        (_day(9, 0), True),  # the start instant is inside
        (_day(12, 0), True),
        (_day(17, 59), True),
        (_day(18, 0), False),  # the end instant is outside
        (_day(23, 0), False),
        (_day(3, 0), False),
    ],
)
def test_a_normal_window_is_half_open(moment: datetime, expected: bool) -> None:
    """Half-open, so 18:00 belongs to exactly one window and not to two."""
    assert OFFICE.contains(moment) is expected


@pytest.mark.parametrize(
    "moment,expected",
    [
        (_day(19, 59), False),
        (_day(20, 0), True),  # the window opens
        (_day(23, 59), True),  # still the same evening
        (_day(0, 0), True),  # across midnight -- the branch that is easy to omit
        (_day(3, 0), True),
        (_day(6, 59), True),
        (_day(7, 0), False),  # closes in the morning
        (_day(12, 0), False),
    ],
)
def test_a_window_that_crosses_midnight_covers_both_sides_of_it(moment: datetime, expected: bool) -> None:
    """20:00-07:00 is two pieces of the clock face, not an empty interval.

    Implemented as `t >= start or t < end`. Written as `start <= t < end` it is
    always False, which does not crash -- it silently refuses every command and
    queues every confirmation until the request times out.
    """
    assert EVENING.crosses_midnight is True
    assert EVENING.contains(moment) is expected


def test_an_equal_start_and_end_means_always_active() -> None:
    """00:00-00:00 is how a config file says 'no restriction'."""
    always = ActivationWindow(time(0, 0), time(0, 0))
    assert always.always_active is True
    assert always.contains(_day(3, 0)) and always.contains(_day(15, 0))
    assert always.describe() == "always"


def test_next_start_is_never_in_the_past() -> None:
    """It anchors a queued confirmation's deadline.

    A next_start in the past would make a queued request expire the instant it
    was released -- the change would be parked and then dropped, which is the
    failure UC3 E2 exists to prevent.
    """
    assert OFFICE.next_start(_day(3, 0)) == _day(9, 0)  # later today
    assert OFFICE.next_start(_day(12, 0)) == _day(9, 0, day=19)  # already passed: tomorrow
    assert OFFICE.next_start(_day(9, 0)) == _day(9, 0)  # exactly at the boundary


def test_next_start_across_midnight_returns_this_evening_not_this_morning() -> None:
    """At 12:00 the 20:00-07:00 window next opens at 20:00 tonight."""
    assert EVENING.next_start(_day(12, 0)) == _day(20, 0)
    # At 03:00 the window is already open, and today's 20:00 is the next opening.
    assert EVENING.next_start(_day(3, 0)) == _day(20, 0)


def test_a_window_parses_from_the_string_a_config_file_would_hold() -> None:
    assert ActivationWindow.parse("20:00-07:00") == EVENING
    assert ActivationWindow.parse(" 09:00 - 18:00 ") == OFFICE
    with pytest.raises(ValueError, match="20:00-07:00"):
        ActivationWindow.parse("evenings")


# -- the agent: commands ----------------------------------------------------


def _calendar_with_flight() -> FakeCalendar:
    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            event_id="E1",
            title="Flight AI440 BLR to MAA",
            start_time=datetime(2026, 8, 20, 9, 15),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
        ).with_alarm_offset(60, "AL-E1")
    )
    return calendar


def test_a_command_outside_the_window_is_refused_and_changes_nothing() -> None:
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar, activation=OFFICE)

    result = agent.handle_command("book a flight to Chennai on 20 August at 9am", _day(3, 0))

    assert result.status == "refused"
    assert "09:00" in result.message
    assert calendar.mutations == []


def test_the_same_command_inside_the_window_is_handled_normally() -> None:
    """The refusal must be about the clock and nothing else."""
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar, activation=OFFICE)

    result = agent.handle_command("book a flight to Chennai on 21 August at 9am", _day(12, 0))

    assert result.status == "success"
    assert [m[0] for m in calendar.mutations] == ["CREATE"]


def test_a_refused_command_still_emits_a_well_formed_trajectory() -> None:
    """FR7: a refusal is behaviour, and Part B has to be able to score it."""
    agent = SyncAgent(FakeCalendar(), activation=OFFICE)
    trajectory = agent.handle_command("cancel my Chennai flight", _day(3, 0)).trajectory

    assert trajectory["outcome"]["status"] == "refused"
    assert trajectory["actions"] == []
    assert trajectory["reasoning"]


def test_a_midnight_crossing_window_lets_a_command_through_at_3am() -> None:
    """The traveller configured evenings; 03:00 is inside their window."""
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar, activation=EVENING)

    result = agent.handle_command("book a flight to Chennai on 21 August at 9am", _day(3, 0))

    assert result.status == "success"
    assert agent.handle_command("book a flight to Delhi on 22 August at 9am", _day(12, 0)).status == "refused"


# -- UC3 E2: the queued confirmation ---------------------------------------


def _cancellation_outside_the_window() -> tuple[FakeCalendar, SyncAgent, object]:
    """A real cancellation email arriving at 03:00, window 09:00-18:00."""
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar, activation=OFFICE)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    result = watcher.scan_inbox(_day(3, 0))[0]
    return calendar, agent, result


def test_a_destructive_confirmation_raised_outside_the_window_is_queued() -> None:
    """UC3 E2. Not applied, not dropped -- parked for the next active window.

    This is the exception the existing `queue_for_active_window` was written for
    and which nothing called until now.
    """
    calendar, agent, result = _cancellation_outside_the_window()
    request = result.confirmation

    assert result.status == "awaiting_confirmation"
    assert request.status is ConfirmationStatus.QUEUED
    assert request.decision.action is ActionType.DELETE
    # The deadline was re-anchored to the window opening by
    # queue_for_active_window, so the user gets their full 15 minutes from 09:00
    # rather than having it expire while they sleep.
    assert request.timeout_at == _day(9, 0) + DEFAULT_TIMEOUT


def test_a_queued_confirmation_deleted_nothing() -> None:
    """The FR5 gate, at the point where it is most tempting to skip it.

    'The user is asleep and the airline says it is cancelled' is exactly the
    argument for auto-applying, and it is exactly what must not happen.
    """
    calendar, _agent, _result = _cancellation_outside_the_window()

    assert calendar.mutations == []
    assert "E1" in calendar.events


def test_a_queued_confirmation_is_not_lost_and_says_when_it_will_be_asked() -> None:
    """The other half: parked is not the same as dropped."""
    _calendar, agent, result = _cancellation_outside_the_window()

    assert agent.confirmations.count() == 1
    assert "09:00" in result.message
    assert "AI440" in result.message


def test_a_queued_confirmation_is_invisible_until_the_window_opens() -> None:
    """A queued request must not be answerable by a stray 'yes' at 04:00.

    It is also why release_queued exists: without it the request stays invisible
    to `pending()` forever and a later 'yes' finds nothing.
    """
    _calendar, agent, _result = _cancellation_outside_the_window()

    assert agent.confirmations.pending(_day(4, 0)) == []
    assert len(release_queued(agent.confirmations, OFFICE, _day(4, 0))) == 0

    released = release_queued(agent.confirmations, OFFICE, _day(9, 5))
    assert [r.status for r in released] == [ConfirmationStatus.PENDING]
    assert len(agent.confirmations.pending(_day(9, 5))) == 1


def test_the_queued_change_is_applied_only_after_the_user_confirms_in_window() -> None:
    """End to end: parked at 03:00, asked at 09:05, applied on consent."""
    calendar, agent, _result = _cancellation_outside_the_window()

    answer = agent.handle_command("yes", _day(9, 5))

    assert answer.status == "success"
    assert calendar.mutations == [("DELETE", "E1")]
    assert calendar.events == {}


def test_declining_a_released_queued_change_leaves_the_calendar_alone() -> None:
    calendar, agent, _result = _cancellation_outside_the_window()

    answer = agent.handle_command("no", _day(9, 5))

    assert answer.status == "refused"
    assert calendar.mutations == []
    assert "E1" in calendar.events


def test_a_yes_arriving_before_the_window_opens_cannot_apply_the_queued_change() -> None:
    """The command gate and the queue have to agree, or one undoes the other."""
    calendar, agent, _result = _cancellation_outside_the_window()

    answer = agent.handle_command("yes", _day(4, 0))

    assert answer.status == "refused"
    assert calendar.mutations == []
    assert agent.confirmations.nth(1).status is ConfirmationStatus.QUEUED


def test_a_repeat_notification_reuses_the_queued_request(tmp_path) -> None:
    """NFR2 has to hold across the window boundary too.

    `ConfirmationStore.find_pending_for` cannot see a QUEUED request, so without
    `find_open_confirmation` the same cancellation arriving twice at 03:00 opens
    two prompts -- the duplicate-prompt bug, moved one status along.
    """
    calendar, agent, first = _cancellation_outside_the_window()
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml", message_id="<resent@airline>")

    second = watcher.scan_inbox(_day(4, 0))[0]

    assert agent.confirmations.count() == 1
    assert second.confirmation.request_id == first.confirmation.request_id
    assert "already" in second.message
    assert calendar.mutations == []


def test_a_queued_request_that_outlived_its_re_anchored_deadline_expires_safely() -> None:
    """UC3 E1 still applies after UC3 E2: a stale yes is not consent.

    Releasing a queued request does not exempt it from the timeout. Once
    released it becomes PENDING, and the very next `pending(now)` expires it if
    its re-anchored deadline has passed -- so a "yes" two hours late deletes
    nothing and the user is asked again rather than obeyed.
    """
    calendar, agent, _result = _cancellation_outside_the_window()

    # The window opened at 09:00 and the deadline was 09:15; it is now 11:00.
    answer = agent.handle_command("yes", _day(11, 0))

    assert answer.status == "refused"
    assert calendar.mutations == []
    assert agent.confirmations.nth(1).status is ConfirmationStatus.TIMED_OUT


def test_queue_outside_window_is_a_no_op_when_the_user_is_reachable() -> None:
    """Inside the window the prompt is delivered, not parked."""
    store = ConfirmationStore()
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar, confirmations=store, activation=OFFICE)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")

    result = watcher.scan_inbox(_day(12, 0))[0]

    assert result.confirmation.status is ConfirmationStatus.PENDING
    assert queue_outside_window(result.confirmation, OFFICE, _day(12, 0)) is False
    assert calendar.mutations == []


def test_find_open_confirmation_matches_a_queued_request_by_event_id() -> None:
    _calendar, agent, result = _cancellation_outside_the_window()
    found = find_open_confirmation(agent.confirmations, None, result.decision.target_event_id, _day(4, 0))
    assert found is not None
    assert found.request_id == result.confirmation.request_id


# -- the default: nothing changes ------------------------------------------


def test_activation_defaults_to_none_and_leaves_every_path_untouched() -> None:
    """The 441 tests that predate Phase 5 all depend on this.

    Asserted directly rather than inferred from the suite passing: an agent that
    silently acquired quiet hours because a parameter appeared would be a
    regression that only shows up at 3am.
    """
    calendar = _calendar_with_flight()
    agent = SyncAgent(calendar)
    assert agent.activation is None

    # 03:00 -- outside any plausible office window, and handled anyway.
    result = agent.handle_command("book a flight to Chennai on 21 August at 9am", _day(3, 0))
    assert result.status == "success"

    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    prompt = watcher.scan_inbox(_day(3, 0))[0]

    assert prompt.status == "awaiting_confirmation"
    # PENDING, not QUEUED: with no window configured the user is always reachable.
    assert prompt.confirmation.status is ConfirmationStatus.PENDING
    assert prompt.confirmation.timeout_at == _day(3, 0) + DEFAULT_TIMEOUT
    assert agent.resolve_confirmation(prompt.confirmation.request_id, True, _day(3, 5)).status == "success"
    assert ("DELETE", "E1") in calendar.mutations


def test_release_and_queue_helpers_are_inert_without_a_window() -> None:
    """activation=None must not merely be untested -- it must be a no-op."""
    store = ConfirmationStore()
    assert release_queued(store, None, _day(3, 0)) == []


def test_the_window_is_configurable_per_traveller() -> None:
    """NFR4 says user-configurable; two agents must be able to disagree."""
    question = "when is my Chennai flight"
    early = SyncAgent(_calendar_with_flight(), activation=ActivationWindow(time(6, 0), time(10, 0)))
    late = SyncAgent(_calendar_with_flight(), activation=EVENING)
    moment = _day(6, 30)

    # 06:30 is inside 06:00-10:00 and inside 20:00-07:00; the same instant is
    # inside both, and 12:00 is inside neither.
    assert early.handle_command(question, moment).status == "success"
    assert late.handle_command(question, moment).status == "success"
    assert early.handle_command(question, _day(12, 0)).status == "refused"
    assert late.handle_command(question, _day(12, 0)).status == "refused"


def test_the_window_never_reads_the_clock_itself() -> None:
    """Every entry point takes `now`, so the scenario suite stays replayable.

    A window that called datetime.now() internally would make the whole NFR4 path
    untestable and every scenario using it flaky at the boundary.
    """
    frozen = _day(3, 0)
    assert OFFICE.contains(frozen) is False
    assert OFFICE.next_start(frozen) == _day(9, 0)
    # Same answers a second time, and a year from now.
    assert OFFICE.contains(frozen) is False
    assert OFFICE.next_start(frozen + timedelta(days=365)) == datetime(2027, 8, 18, 9, 0)
