"""Live walkthrough of everything built in Phases 0-3.

    python -m packages.demo              
    python -m packages.demo --only 4    
    python -m packages.demo --list

Nothing here is staged: every number is computed and every calendar shown is the
real state of a FakeCalendar after the agent has acted on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta
from email import message_from_string
from pathlib import Path

from packages.bench.scenarios import ScenarioRunner, load_scenarios, run_all
from packages.core.confirmation import ConfirmationStore
from packages.core.decisions import ActionType, Decision, Reason
from packages.core.models import BookingType, CalendarEvent
from packages.nlp.baseline import BaselineExtractor
from packages.nlp.dataset import read_jsonl
from packages.nlp.generate import DATA_DIR
from packages.nlp.metrics import evaluate, safety_critical_confusions
from packages.nlp.resolve import resolve_datetime
from packages.providers.fake_calendar import FakeCalendar
from packages.providers.fake_inbox import FakeInbox
from packages.sync_agent.agent import SyncAgent
from packages.sync_agent.email_watcher import EmailIngestionModule
from packages.sync_agent.reconciliation import ReconciliationEngine

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "emails" / "fixtures"
NOW = datetime(2026, 8, 18, 8, 0)

W = 78


def head(number: int, title: str, subtitle: str = "") -> None:
    print("\n" + "=" * W)
    print("  %d. %s" % (number, title.upper()))
    if subtitle:
        print("     %s" % subtitle)
    print("=" * W)


def step(text: str) -> None:
    print("\n  > %s" % text)


def out(text: str, indent: int = 6) -> None:
    print(" " * indent + text)


def calendar_state(calendar: FakeCalendar, label: str = "calendar") -> None:
    events = calendar.list_events()
    if not events:
        out("%-10s (empty)" % (label + ":"))
        return
    for i, event in enumerate(events):
        prefix = label + ":" if i == 0 else ""
        alarm = ("  alarm %s" % event.alarm.trigger_time.strftime("%d %b %H:%M")) if event.alarm else ""
        out("%-10s %s  %s%s" % (prefix, event.start_time.strftime("%d %b %H:%M"), event.title, alarm))


def _fixture(name: str) -> tuple[str, str]:
    msg = message_from_string((FIXTURES / name).read_text(encoding="utf-8"))
    return msg.get("Subject", ""), str(msg.get_payload())


# ---------------------------------------------------------------- 1 ----------


def demo_safety_gate() -> None:
    head(1, "The FR5 safety gate", "Structural, not a policy check (Phase 0-1)")

    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent("E1", "Flight AI440 BLR to MAA", datetime(2026, 8, 20, 9, 15), booking_ref="AI440-XY12")
    )
    engine = ReconciliationEngine(calendar)
    delete = Decision(
        action=ActionType.DELETE, reason=Reason.CANCELLATION, target_event_id="E1", payload=calendar.events["E1"]
    )

    step("A rogue caller tries to delete the event directly:")
    out("engine.apply(delete_decision)")
    try:
        engine.apply(delete)
        out("!! DELETED -- the gate failed")
    except PermissionError as exc:
        out("PermissionError: %s" % str(exc).split("(")[0].strip())
    calendar_state(calendar)

    step("It tries to forge the consent token instead:")
    out("ConfirmedDecision(delete_decision, 'CR-fake', now)")
    from packages.core.decisions import ConfirmedDecision

    try:
        ConfirmedDecision(delete, "CR-fake", NOW)
        out("!! FORGED -- the gate failed")
    except PermissionError:
        out("PermissionError: only ConfirmationRequest.resolve() can create one")

    step("The legitimate path: ask, then act.")
    store = ConfirmationStore()
    request = store.open(delete, NOW)
    out('prompt -> "%s"' % request.prompt)
    token = request.confirm(NOW)
    engine.apply_destructive(token)
    out("user confirmed -> apply_destructive(token)")
    calendar_state(calendar, "after")
    out("mutations: %s" % calendar.mutations)


# ---------------------------------------------------------------- 2 ----------


def demo_scenarios() -> None:
    head(2, "Scenario suite", "66 scenarios / 10 families, the FR10 benchmark and the FR5 build gate")

    results = run_all()
    families = Counter(r.family for r in results)
    passed = sum(1 for r in results if r.passed)
    violations = sum(r.destructive_without_confirm for r in results)

    for family, count in sorted(families.items(), key=lambda kv: -kv[1]):
        family_results = [r for r in results if r.family == family]
        ok = sum(1 for r in family_results if r.passed)
        out("%-24s %d/%d passed" % (family, ok, count))

    print()
    out("total            : %d scenarios, %d passed, %d failed" % (len(results), passed, len(results) - passed))
    out("FR5 violations   : %d   (unconfirmed destructive actions -- gate fails the build above 0)" % violations)

    step("S-401 is the design document's sequence diagram, replayed:")
    s401 = next(s for s in load_scenarios() if s["id"] == "S-401")
    result = ScenarioRunner(s401).run()
    for entry in result.trajectory["reasoning"]:
        out("reasoning %d: %s" % (entry["step"], entry["text"]))
    for call in result.trajectory["tool_calls"]:
        out("tool      : %s" % call["name"])
    for action in result.trajectory["actions"]:
        out(
            "action    : %s %s  confirmed_by_user=%s"
            % (action["type"], action["target"], action["confirmed_by_user"])
        )


# ---------------------------------------------------------------- 3 ----------


def demo_nlu() -> None:
    head(3, "NLU pipeline (D1)", "Leakage-safe splits, and the metric the build is held to")

    splits = {name: read_jsonl(DATA_DIR / "splits" / ("%s.jsonl" % name)) for name in ("train", "dev", "test")}
    out("splits: " + ", ".join("%s %d" % (k, len(v)) for k, v in splits.items()))
    families = {name: {r.template_id for r in rows} for name, rows in splits.items()}
    out(
        "template families shared between train and test: %d  <- leakage check"
        % len(families["train"] & families["test"])
    )

    extractor = BaselineExtractor()
    print()
    for name in ("dev", "test"):
        gold = splits[name]
        predictions = [extractor.extract(u.text) for u in gold]
        report = evaluate(gold, predictions)
        unsafe = safety_critical_confusions(gold, predictions)
        label = "%s (tuned on)" % name if name == "dev" else "%s (HELD OUT)" % name
        out(
            "%-16s intent %.3f   slot-F1 %.3f   frame %.3f   safety-critical %d"
            % (label, report.intent_accuracy, report.slot_micro.f1, report.frame_accuracy, unsafe)
        )
    out("")
    out("The gap between the two rows is why the split is made over template")
    out("families rather than rows. Quote the held-out number.")

    step("Extraction on a live utterance:")
    text = "shift my Chennai flight to half past eight"
    frame = extractor.extract(text)
    out('"%s"' % text)
    out("intent: %s" % frame.intent.value)
    for slot in frame.slots:
        out("slot  : %-13s %r" % (slot.name, slot.value))

    step("Resolution refuses to guess:")
    for date_text, time_text in [("20 August", "half past eight"), ("next Friday", None), (None, "9:15 am"), ("20 August", "at 7")]:
        resolution = resolve_datetime(date_text, time_text, NOW)
        rendered = resolution.value.strftime("%d %b %Y %H:%M") if resolution.ok else "REFUSED"
        out("date=%-12r time=%-18r -> %-18s %s" % (date_text, time_text, rendered, resolution.note or ""))


# ---------------------------------------------------------------- 4 ----------


def demo_email_flow() -> None:
    head(4, "The sequence diagram, end to end", "Phase 3: from a real .eml file to a deleted event")

    calendar = FakeCalendar()
    calendar.seed(
        CalendarEvent(
            "E1",
            "Flight AI440 BLR to MAA",
            datetime(2026, 8, 20, 9, 15),
            booking_ref="AI440-XY12",
            booking_type=BookingType.FLIGHT,
        ).with_alarm_offset(60, "AL-E1")
    )
    agent = SyncAgent(calendar)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent)

    calendar_state(calendar, "before")

    step("A cancellation email arrives (data/emails/fixtures/cancel_ai440.eml):")
    subject, body = _fixture("cancel_ai440.eml")
    out("Subject: %s" % subject)
    for line in [line for line in body.splitlines() if line.strip()][2:5]:
        out(line.strip())

    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    result = watcher.scan_inbox(NOW)[0]

    step("The agent extracts, reconciles, and STOPS at the gate:")
    out("status  : %s" % result.status)
    out('prompt  : "%s"' % result.message)
    calendar_state(calendar, "calendar")
    out("mutations so far: %s   <- nothing touched yet" % (calendar.mutations or "none"))

    step('The traveller answers "yes go ahead":')
    confirmed = agent.handle_command("yes go ahead", NOW + timedelta(minutes=3))
    out("status  : %s" % confirmed.status)
    out('reply   : "%s"' % confirmed.message)
    calendar_state(calendar, "after")
    out("mutations: %s" % calendar.mutations)

    step("The trajectory this produced (FR7 contract, Part B consumes this):")
    trajectory = confirmed.trajectory
    out(json.dumps({k: trajectory[k] for k in ("archetype", "task", "outcome")}, indent=2).replace("\n", "\n      "))
    for action in trajectory["actions"]:
        out("action: %s" % json.dumps(action))


# ---------------------------------------------------------------- 5 ----------


def demo_safety_behaviours() -> None:
    head(5, "What the agent refuses to do", "Every case here is a way an agent damages real data")

    def fresh() -> tuple[FakeCalendar, SyncAgent]:
        calendar = FakeCalendar()
        calendar.seed(
            CalendarEvent(
                "E1",
                "my Chennai flight",
                datetime(2026, 8, 20, 9, 15),
                booking_ref="AI440-XY12",
                booking_type=BookingType.FLIGHT,
            )
        )
        return calendar, SyncAgent(calendar)

    step("A promotional email whose subject says 'cancelled':")
    calendar, agent = fresh()
    subject, _ = _fixture("promo_fare_sale.eml")
    out('Subject: "%s"' % subject)
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "promo_fare_sale.eml")
    result = EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW)[0]
    out("-> %s: %s" % (result.status, result.message))
    out("-> prompts raised: %d, mutations: %s" % (agent.confirmations.count(), calendar.mutations or "none"))

    step("A stray 'yes' with no question behind it:")
    calendar, agent = fresh()
    result = agent.handle_command("yes go ahead", NOW)
    out("-> %s: %s" % (result.status, result.message))

    step("A self-corrected command, while a cancellation is pending:")
    calendar, agent = fresh()
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "cancel_ai440.eml")
    EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW)
    result = agent.handle_command("no wait, don't cancel it", NOW + timedelta(minutes=1))
    out('"no wait, don\'t cancel it"')
    out("-> %s: %s" % (result.status, result.message))
    calendar_state(calendar, "calendar")

    step("A booking command with no date:")
    calendar, agent = fresh()
    result = agent.handle_command("book me a flight to Chennai", NOW)
    out("-> %s: %s" % (result.status, result.message))

    step("An ambiguous reference (two matching flights):")
    calendar = FakeCalendar()
    for event_id, hour in (("E1", 9), ("E2", 19)):
        calendar.seed(
            CalendarEvent(event_id, "my Chennai flight", datetime(2026, 8, 20, hour, 15), booking_type=BookingType.FLIGHT)
        )
    agent = SyncAgent(calendar)
    result = agent.handle_command("cancel my Chennai flight", NOW)
    out("-> %s: %s" % (result.status, result.message))
    out("-> mutations: %s" % (calendar.mutations or "none"))

    step("An unreadable booking email (quarantined, not dropped):")
    calendar, agent = fresh()
    inbox = FakeInbox()
    inbox.deliver_fixture(FIXTURES / "unparseable_itinerary.eml")
    result = EmailIngestionModule(mail=inbox, agent=agent).scan_inbox(NOW)[0]
    out("-> %s: %s" % (result.status, result.message))
    out("-> quarantined for review: %s" % agent.quarantine)


# ---------------------------------------------------------------- 6 ----------


def demo_happy_paths() -> None:
    head(6, "What the agent does do", "Voice booking, idempotency, reschedule, and status queries")

    calendar = FakeCalendar()
    agent = SyncAgent(calendar)

    step('Voice: "book me a flight to Chennai on 20 August"')
    result = agent.handle_command("book me a flight to Chennai on 20 August", NOW)
    out("-> %s: %s" % (result.status, result.message))
    calendar_state(calendar)

    step("The same confirmation email arrives twice (NFR2):")
    calendar2 = FakeCalendar()
    agent2 = SyncAgent(calendar2)
    inbox = FakeInbox()
    watcher = EmailIngestionModule(mail=inbox, agent=agent2)
    inbox.deliver_fixture(FIXTURES / "confirm_ai440.eml", message_id="m1")
    watcher.scan_inbox(NOW)
    inbox.deliver_fixture(FIXTURES / "confirm_ai440.eml", message_id="m2")
    second = watcher.scan_inbox(NOW + timedelta(minutes=30))[0]
    out("second delivery -> %s: %s" % (second.status, second.message))
    out("calendar writes: %d   <- one booking, one write" % len(calendar2.mutations))
    calendar_state(calendar2)

    step("A reschedule email, gated then applied:")
    inbox.deliver_fixture(FIXTURES / "reschedule_ai440.eml", message_id="m3")
    prompt = watcher.scan_inbox(NOW + timedelta(hours=1))[0]
    out('prompt -> "%s"' % prompt.message)
    agent2.resolve_confirmation(prompt.confirmation.request_id, True, NOW + timedelta(hours=1, minutes=1))
    calendar_state(calendar2, "after")

    step("Status queries (FR6) -- read-only:")
    before = len(calendar2.mutations)
    actions = 0
    for question in ["when is my flight", "do I have anything booked on 20 August", "when is my hotel booking"]:
        answer = agent2.handle_command(question, NOW)
        actions += len(answer.trajectory["actions"])
        out('"%s"' % question)
        out("   -> %s" % answer.message)
    out("")
    out("calendar writes caused by all three questions : %d" % (len(calendar2.mutations) - before))
    out("actions recorded in their trajectories        : %d   <- a question is not an instruction" % actions)


SECTIONS = [
    ("The FR5 safety gate", demo_safety_gate),
    ("Scenario suite", demo_scenarios),
    ("NLU pipeline", demo_nlu),
    ("Sequence diagram end to end", demo_email_flow),
    ("What the agent refuses", demo_safety_behaviours),
    ("What the agent does", demo_happy_paths),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="packages.demo")
    parser.add_argument("--only", type=int, metavar="N", help="run one section")
    parser.add_argument("--list", action="store_true", help="list sections")
    args = parser.parse_args(argv)

    if args.list:
        for i, (title, _) in enumerate(SECTIONS, start=1):
            print("%d. %s" % (i, title))
        return 0

    print("\n" + "#" * W)
    print("#  AI BOOKING-LIFECYCLE SYNC ASSISTANT -- Phases 0-3 demo")
    print("#  Every figure below is computed live; nothing is transcribed.")
    print("#" * W)

    chosen = [SECTIONS[args.only - 1]] if args.only else SECTIONS
    for _, fn in chosen:
        fn()

    print("\n" + "=" * W)
    print("  END OF DEMO")
    print("=" * W + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
