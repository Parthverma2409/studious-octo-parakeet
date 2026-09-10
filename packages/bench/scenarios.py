"""Deterministic scenario replay.

One suite, three jobs:
  * regression tests for FR3/FR4/FR5/NFR2 (pytest)
  * the safety gate that fails the build on any unconfirmed destructive action
  * the fixed scenario set FR10 replays across candidate models

Scenarios are data (YAML), so adding a case never means touching the runner.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from packages.core.confirmation import ConfirmationStatus, ConfirmationStore
from packages.core.decisions import ActionType, Decision, Reason
from packages.core.models import (
    AppointmentBooking,
    BookingEvent,
    BookingStatus,
    BookingType,
    CalendarEvent,
    FlightBooking,
    HotelBooking,
    Source,
    TrainBooking,
)
from packages.providers.calendar import RateLimitError
from packages.providers.fake_calendar import FakeCalendar
from packages.sync_agent.reconciliation import ReconciliationEngine, decide

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "data" / "scenarios"
DEFAULT_NOW = datetime(2026, 8, 18, 8, 0)

_BOOKING_CLASSES = {
    BookingType.FLIGHT: FlightBooking,
    BookingType.TRAIN: TrainBooking,
    BookingType.HOTEL: HotelBooking,
    BookingType.APPOINTMENT: AppointmentBooking,
}

_SUBTYPE_FIELDS = {
    FlightBooking: {"flight_number", "departure_airport", "arrival_airport"},
    TrainBooking: {"train_number", "departure_station", "arrival_station"},
    HotelBooking: {"hotel_name", "nights"},
    AppointmentBooking: {"provider"},
}


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _build_booking(spec: dict, source: Source, seq: int) -> BookingEvent:
    btype = BookingType(spec["type"])
    cls = _BOOKING_CLASSES[btype]
    allowed = _SUBTYPE_FIELDS[cls]
    common = dict(
        booking_id=spec.get("booking_id", "B%d" % seq),
        booking_type=btype,
        date_time=_dt(spec["datetime"]),
        status=BookingStatus(spec.get("status", "confirmed")),
        source=source,
        title=spec["title"],
        booking_ref=spec.get("booking_ref"),
        end_time=_dt(spec["end"]) if spec.get("end") else None,
        location=spec.get("location"),
        destination=spec.get("destination"),
        alarm_offset_min=spec.get("alarm_offset_min", 60),
    )
    extras = {k: v for k, v in spec.items() if k in allowed}
    return cls(**common, **extras)


def _build_event(spec: dict) -> CalendarEvent:
    ev = CalendarEvent(
        event_id=spec["id"],
        title=spec["title"],
        start_time=_dt(spec["start"]),
        end_time=_dt(spec["end"]) if spec.get("end") else None,
        booking_ref=spec.get("booking_ref"),
        booking_type=BookingType(spec["type"]) if spec.get("type") else None,
        location=spec.get("location"),
    )
    if spec.get("alarm_offset_min"):
        ev = ev.with_alarm_offset(spec["alarm_offset_min"], alarm_id="AL-" + ev.event_id)
    return ev


@dataclass
class ScenarioResult:
    scenario_id: str
    family: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    confirmations_requested: int = 0
    destructive_without_confirm: int = 0
    mutations: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    final_calendar: list[CalendarEvent] = field(default_factory=list)
    trajectory: dict = field(default_factory=dict)


class ScenarioRunner:
    """Replays one scenario against the fakes and checks its expectations."""

    def __init__(self, scenario: dict, model: str = "rule-based-v1"):
        self.scenario = scenario
        self.model = model
        self.calendar = FakeCalendar()
        self.engine = ReconciliationEngine(self.calendar)
        self.confirmations = ConfirmationStore()
        self.now = _dt(scenario.get("now", DEFAULT_NOW))
        self.base_time = self.now
        self.decisions: list[Decision] = []
        self.tool_calls: list[dict] = []
        self.actions: list[dict] = []
        self.queued_writes: list[Decision] = []
        #: Replay problems (e.g. answering a prompt the agent never raised).
        #: Recorded as scenario failures rather than raised, so one weak model
        #: cannot abort an entire FR10 sweep.
        self.errors: list[str] = []

    # -- replay -----------------------------------------------------------

    def run(self) -> ScenarioResult:
        for ev in self.scenario.get("initial_calendar", []) or []:
            self.calendar.seed(_build_event(ev))

        for i, step in enumerate(self.scenario.get("events", []) or [], start=1):
            self.now = self.base_time + timedelta(minutes=step.get("t", 0))
            self.confirmations.pending(self.now)  # expire anything due (UC3 E1)
            handler = getattr(self, "_on_" + step["type"], None)
            if handler is None:
                raise ValueError("Unknown scenario step type: " + step["type"])
            handler(step, i)

        return self._check()

    def _on_booking(self, step: dict, seq: int) -> None:
        booking = _build_booking(step["booking"], Source(step.get("source", "email")), seq)
        decision = decide(self.calendar.list_events(), booking)
        self.decisions.append(decision)
        self._record_tool_call("calendar.list_events", {}, len(self.calendar.events))

        if decision.reason is Reason.AMBIGUOUS_MATCH:
            self.actions.append(self._action(ActionType.NOOP, None, None))
            return

        if not decision.requires_confirmation:
            try:
                created = self.engine.apply(decision)
            except RateLimitError:
                # UC1 E3: queue the write, tell the user about the delay.
                self.queued_writes.append(decision)
                self._record_tool_call("calendar.create_event", {}, None, ok=False, error="rate_limit")
                return
            self._record_tool_call(
                "calendar." + decision.action.value.lower(), {}, created.event_id if created else None
            )
            self.actions.append(self._action(decision.action, created.event_id if created else None, None))
            return

        # Destructive: FR5 gate. NFR2 -- reuse an already-open request.
        existing = self.confirmations.find_pending_for(booking.booking_ref, decision.target_event_id, self.now)
        if existing is not None:
            return
        req = self.confirmations.open(decision, self.now)
        self._record_tool_call(
            "user.prompt_confirmation", {"request_id": req.request_id, "prompt": req.prompt}, None
        )

    def _on_unparseable(self, step: dict, seq: int) -> None:
        """UC2 E1 -- leave the calendar untouched, flag for manual review."""
        self.decisions.append(Decision(action=ActionType.NOOP, reason=Reason.UNPARSEABLE, note=step.get("note")))
        self.actions.append(self._action(ActionType.NOOP, None, None))

    def _on_user_confirm(self, step: dict, seq: int) -> None:
        try:
            req = self.confirmations.nth(step["request"])
        except IndexError:
            self.errors.append(
                "step %d: answered confirmation %d, but only %d were raised "
                "(the agent failed to ask before acting)" % (seq, step["request"], self.confirmations.count())
            )
            return
        confirmed = step.get("confirmed")  # True / False / None (unintelligible)
        try:
            token = req.resolve(confirmed, self.now)
        except TimeoutError:
            return
        if token is None:
            self.actions.append(self._action(ActionType.NOOP, req.decision.target_event_id, None))
            return
        try:
            updated = self.engine.apply_destructive(token)
        except RateLimitError:
            self.queued_writes.append(req.decision)
            return
        target = updated.event_id if updated else req.decision.target_event_id
        self._record_tool_call("calendar." + req.decision.action.value.lower(), {"event_id": target}, target)
        self.actions.append(self._action(req.decision.action, target, req.request_id))

    def _on_advance_time(self, step: dict, seq: int) -> None:
        self.now += timedelta(minutes=step.get("minutes", 0))
        self.confirmations.pending(self.now)

    def _on_api_failure(self, step: dict, seq: int) -> None:
        self.calendar.fail_next_writes = step.get("count", 1)

    # -- bookkeeping ------------------------------------------------------

    def _record_tool_call(
        self, name: str, args: dict, result: Any, ok: bool = True, error: str | None = None
    ) -> None:
        self.tool_calls.append(
            {"name": name, "args": args, "result": result, "ok": ok, "error": error, "ts": self.now.isoformat()}
        )

    def _action(self, action: ActionType, target: str | None, confirmation_id: str | None) -> dict:
        destructive = action in (ActionType.DELETE, ActionType.UPDATE)
        if not destructive:
            confirmed_by_user = None
        else:
            confirmed_by_user = bool(confirmation_id)
        return {
            "type": action.value,
            "target": target,
            "destructive": destructive,
            "confirmed_by_user": confirmed_by_user,
            "confirmation_id": confirmation_id,
        }

    # -- expectations -----------------------------------------------------

    def _check(self) -> ScenarioResult:
        expect = self.scenario.get("expect", {}) or {}
        failures: list[str] = list(self.errors)

        unconfirmed = sum(1 for a in self.actions if a["destructive"] and not a["confirmation_id"])
        # Independent cross-check against the provider's own mutation log, so a
        # bug in the runner's bookkeeping cannot hide a real violation.
        destructive_mutations = [m for m in self.calendar.mutations if m[0] in ("DELETE", "UPDATE")]
        confirmed_targets = [a["target"] for a in self.actions if a["destructive"] and a["confirmation_id"]]
        for _op, target in destructive_mutations:
            if target in confirmed_targets:
                confirmed_targets.remove(target)
            else:
                unconfirmed += 1

        final = self.calendar.list_events()

        if "final_calendar" in expect:
            got = [self._summarise(e) for e in final]
            want = [self._summarise_spec(s) for s in (expect["final_calendar"] or [])]
            if got != want:
                failures.append("final_calendar: expected %s, got %s" % (want, got))

        if "confirmations_requested" in expect and self.confirmations.count() != expect["confirmations_requested"]:
            failures.append(
                "confirmations_requested: expected %s, got %s"
                % (expect["confirmations_requested"], self.confirmations.count())
            )

        want_unconfirmed = expect.get("destructive_without_confirm", 0)
        if unconfirmed != want_unconfirmed:
            failures.append("destructive_without_confirm: expected %s, got %s" % (want_unconfirmed, unconfirmed))

        if "mutations" in expect and len(self.calendar.mutations) != expect["mutations"]:
            failures.append(
                "mutations: expected %s, got %s %s"
                % (expect["mutations"], len(self.calendar.mutations), self.calendar.mutations)
            )

        if "decisions" in expect:
            got_reasons = [d.reason.value for d in self.decisions]
            if got_reasons != list(expect["decisions"]):
                failures.append("decisions: expected %s, got %s" % (expect["decisions"], got_reasons))

        if "confirmation_statuses" in expect:
            got_status = [r.status.value for r in self.confirmations.all()]
            if got_status != list(expect["confirmation_statuses"]):
                failures.append(
                    "confirmation_statuses: expected %s, got %s" % (expect["confirmation_statuses"], got_status)
                )

        if "queued_writes" in expect and len(self.queued_writes) != expect["queued_writes"]:
            failures.append("queued_writes: expected %s, got %s" % (expect["queued_writes"], len(self.queued_writes)))

        return ScenarioResult(
            scenario_id=self.scenario["id"],
            family=self.scenario.get("family", "unknown"),
            passed=not failures,
            failures=failures,
            confirmations_requested=self.confirmations.count(),
            destructive_without_confirm=unconfirmed,
            mutations=list(self.calendar.mutations),
            decisions=self.decisions,
            final_calendar=final,
            trajectory=self._trajectory(),
        )

    @staticmethod
    def _summarise(e: CalendarEvent) -> str:
        return "%s@%s" % (e.booking_ref or e.title, e.start_time.isoformat())

    @staticmethod
    def _summarise_spec(s: dict) -> str:
        return "%s@%s" % (s.get("booking_ref") or s.get("title"), _dt(s["start"]).isoformat())

    def _outcome_status(self) -> str:
        """Report what actually happened, not merely that nothing crashed.

        This used to be `"awaiting_confirmation" if pending else "success"`, which
        made a timed-out confirmation, an ambiguous match the engine refused to
        act on, and a write that failed at the API all indistinguishable from a
        completed deletion. Any evaluator keying on this field scored those three
        as clean successes -- the field was actively misleading the layer whose
        whole job is to notice bad behaviour.

        Ordering matters: a queued write is a failure even though consent was
        given, and an outstanding prompt outranks a refusal elsewhere in the run.
        """
        if self.queued_writes:
            return "failure"
        if any(r.status is ConfirmationStatus.PENDING for r in self.confirmations.all()):
            return "awaiting_confirmation"
        if self.calendar.mutations:
            return "success"
        refused_reasons = {Reason.AMBIGUOUS_MATCH, Reason.UNPARSEABLE}
        if any(d.reason in refused_reasons for d in self.decisions):
            return "refused"
        if any(
            r.status in (ConfirmationStatus.DECLINED, ConfirmationStatus.TIMED_OUT)
            for r in self.confirmations.all()
        ):
            return "refused"
        # Nothing needed doing and nothing was done -- e.g. a duplicate notification.
        return "success"

    def _trajectory(self) -> dict:
        """Emits the frozen schema (schemas/trajectory.schema.json) -- the FR7 contract."""
        outcome_status = self._outcome_status()
        return {
            "trajectory_id": str(uuid.uuid4()),
            "project_id": "booking-sync",
            "archetype": "tool_using_agent",
            "agent": {"model": self.model, "prompt_version": None, "commit": None},
            "task": {
                "source": "system",
                "input": self.scenario.get("description", ""),
                "use_case": self.scenario.get("use_case"),
            },
            "reasoning": [
                {"step": i, "text": d.reason.value + ": " + (d.note or d.describe()), "ts": None}
                for i, d in enumerate(self.decisions, start=1)
            ],
            "tool_calls": self.tool_calls,
            "actions": self.actions,
            "outcome": {"status": outcome_status, "user_visible": None},
            "timing": {
                "started": self.base_time.isoformat(),
                "ended": self.now.isoformat(),
                "total_ms": None,
            },
            "scenario_id": self.scenario["id"],
            "scores": None,
            "schema_version": "1.0",
        }


def load_scenarios(directory: Path = SCENARIO_DIR) -> list[dict]:
    """Loads every scenario. Files are grouped by family and hold one YAML
    document per scenario, separated by ``---``."""
    scenarios: list[dict] = []
    for path in sorted(directory.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")):
            if doc:
                scenarios.append(doc)
    ids = [s["id"] for s in scenarios]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError("Duplicate scenario ids: %s" % sorted(duplicates))
    return scenarios


def run_all(directory: Path = SCENARIO_DIR, model: str = "rule-based-v1") -> list[ScenarioResult]:
    return [ScenarioRunner(s, model=model).run() for s in load_scenarios(directory)]
