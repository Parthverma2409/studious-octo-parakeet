"""SyncAgent -- the orchestrator (UC1, UC2, UC3, UC6-query).

Wires NLP -> ReconciliationEngine -> CalendarProvider, with the FR5 confirmation
gate in the middle. Everything it depends on is a protocol: swap `FakeCalendar`
for Google, or the baseline extractor for Ollama, without touching this file.

It emits a trajectory per interaction in the frozen schema, which is what the
Phase 4 capture SDK will forward rather than reinvent.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from packages.core.confirmation import ConfirmationRequest, ConfirmationStatus, ConfirmationStore
from packages.core.decisions import ActionType, Decision, Reason
from packages.core.matching import find_by_phrase
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
    Traveller,
)
from packages.nlp.baseline import BaselineExtractor
from packages.nlp.email_extract import extract_email
from packages.nlp.extractor import Extractor
from packages.nlp.resolve import resolve_datetime
from packages.nlp.schema import Frame, Intent, RESPONSE_INTENTS
from packages.providers.calendar import CalendarProvider, RateLimitError
from packages.providers.mail import RawEmail
from packages.sync_agent.activation import (
    ActivationWindow,
    find_open_confirmation,
    queue_outside_window,
    release_queued,
)
from packages.sync_agent.reconciliation import ReconciliationEngine, decide

_BOOKING_CLASSES = {
    BookingType.FLIGHT: FlightBooking,
    BookingType.TRAIN: TrainBooking,
    BookingType.HOTEL: HotelBooking,
    BookingType.APPOINTMENT: AppointmentBooking,
}

_INTENT_STATUS = {
    Intent.CREATE_BOOKING: BookingStatus.CONFIRMED,
    Intent.RESCHEDULE_BOOKING: BookingStatus.RESCHEDULED,
    Intent.CANCEL_BOOKING: BookingStatus.CANCELLED,
}


@dataclass
class AgentResult:
    """What the agent did, and what to say to the user.

    `status` mirrors the trajectory schema's outcome vocabulary so the two never
    drift: success, refused, awaiting_confirmation, failure, error.
    """

    status: str
    message: str
    decision: Decision | None = None
    confirmation: ConfirmationRequest | None = None
    needs_clarification: bool = False
    trajectory: dict = field(default_factory=dict)


class SyncAgent:
    def __init__(
        self,
        calendar: CalendarProvider,
        extractor: Extractor | None = None,
        traveller: Traveller | None = None,
        confirmations: ConfirmationStore | None = None,
        activation: ActivationWindow | None = None,
    ):
        self.calendar = calendar
        self.extractor = extractor or BaselineExtractor()
        self.traveller = traveller or Traveller("u1", "Traveller", "traveller@example.com")
        self.confirmations = confirmations or ConfirmationStore()
        self.engine = ReconciliationEngine(calendar)
        self.quarantine: list[dict] = []
        #: NFR4. None means always active, which is the pre-Phase-5 behaviour and
        #: the default: an agent that silently acquires quiet hours because a
        #: parameter appeared is a worse surprise than one that never sleeps.
        self.activation = activation

    # -- UC1 / UC3 / query: a spoken or typed command ---------------------

    def handle_command(self, text: str, now: datetime) -> AgentResult:
        tape = _Tape(source="voice", text=text, now=now, model=self.extractor.name)

        # NFR4. Checked before extraction: outside the window there is no answer
        # worth computing, and nothing the agent could do with one.
        refusal = self._outside_window(now, tape)
        if refusal is not None:
            return refusal

        frame = self.extractor.extract(text)
        tape.tool("nlp.extract", {"text": text}, frame.intent.value)
        tape.think("intent=%s slots=%s" % (frame.intent.value, [(s.name, s.value) for s in frame.slots]))

        if frame.intent in RESPONSE_INTENTS:
            return self._answer_pending(frame, now, tape)

        if frame.intent is Intent.OUT_OF_SCOPE:
            return tape.done(
                AgentResult("refused", "That does not look like a booking request, so I have not changed anything.")
            )

        if frame.intent is Intent.QUERY_STATUS:
            return self._answer_query(frame, now, tape)

        return self._reconcile_from_frame(frame, Source.VOICE, now, tape)

    # -- UC2: an inbound email --------------------------------------------

    def handle_email(self, email: RawEmail, now: datetime) -> AgentResult:
        tape = _Tape(source="email", text=email.subject, now=now, model="rule-based-email-v1")
        # NFR4 governs when the user can be *asked*, not when mail can be read:
        # an airline sends a cancellation at 03:00 whatever the window says, and
        # refusing to read it would lose the change. So email is never refused
        # here -- only the confirmation prompt is deferred (see _decide_and_act).
        self._release_queued(now, tape)
        extraction = extract_email(email.subject, email.body, now)
        tape.tool("email.extract", {"message_id": email.message_id}, extraction.reason or "actionable")

        if not extraction.is_booking_related:
            tape.think("not a booking email (%s)" % extraction.reason)
            return tape.done(AgentResult("refused", "Ignored a non-booking email (%s)." % extraction.reason))

        if not extraction.is_actionable:
            # UC2 E1 -- quarantined and surfaced, never silently dropped.
            self.quarantine.append({"message_id": email.message_id, "subject": email.subject, "reason": extraction.reason})
            tape.think("quarantined: %s" % extraction.reason)
            return tape.done(
                AgentResult("failure", "Could not read this booking email; flagged it for review (%s)." % extraction.reason)
            )

        booking = _build_booking(
            booking_type=extraction.booking_type or BookingType.APPOINTMENT,
            status=extraction.status,
            when=extraction.date_time,
            title=extraction.title,
            source=Source.EMAIL,
            booking_ref=extraction.booking_ref,
        )
        tape.think("email says %s for %s at %s" % (extraction.status.value, extraction.title, extraction.date_time))
        return self._decide_and_act(booking, now, tape)

    # -- UC3: resolving a confirmation ------------------------------------

    def resolve_confirmation(self, request_id: str, confirmed: bool | None, now: datetime) -> AgentResult:
        tape = _Tape(source="voice", text="confirmation:%s" % request_id, now=now, model=self.extractor.name)
        try:
            request = self.confirmations.get(request_id)
        except KeyError:
            return tape.done(AgentResult("error", "No such confirmation request."))
        return self._apply_confirmation(request, confirmed, now, tape)

    def _answer_query(self, frame: Frame, now: datetime, tape: "_Tape") -> AgentResult:
        """FR6 -- answer a status question from reconciled calendar state.

        Strictly read-only. A question is not an instruction, so no path out of
        here touches the calendar or opens a confirmation.
        """
        slots = {s.name: s.value for s in frame.slots}
        events = self.calendar.list_events()
        tape.tool("calendar.list_events", {}, len(events))

        if not events:
            return tape.done(AgentResult("success", "You have nothing booked at the moment."))

        matches = self._candidates_for(slots, events)

        if not matches and slots.get("date"):
            wanted = resolve_datetime(slots["date"], None, now)
            if wanted.ok:
                matches = [e for e in events if e.start_time.date() == wanted.value.date()]
                if not matches:
                    return tape.done(
                        AgentResult("success", "Nothing is booked on %s." % wanted.value.strftime("%d %b"))
                    )

        if not matches and slots.get("destination"):
            matches = find_by_phrase(events, slots["destination"])

        if not matches:
            tape.think("no calendar event matches the query")
            return tape.done(
                AgentResult(
                    "success",
                    "I could not find a booking matching that. You currently have: %s." % _summarise(events),
                )
            )

        tape.think("query matched %d event(s)" % len(matches))
        return tape.done(AgentResult("success", _summarise(matches) + "."))

    def _answer_pending(self, frame: Frame, now: datetime, tape: "_Tape") -> AgentResult:
        pending = self.confirmations.pending(now)
        if not pending:
            # Nothing was asked, so there is nothing to consent to. A stray "yes"
            # must never find something to apply itself to.
            return tape.done(AgentResult("refused", "There is nothing waiting for your confirmation."))
        if len(pending) > 1:
            return tape.done(
                AgentResult(
                    "refused",
                    "There are %d changes waiting. Which one do you mean? %s"
                    % (len(pending), "; ".join(r.prompt for r in pending)),
                    needs_clarification=True,
                )
            )
        confirmed = frame.intent is Intent.CONFIRM_ACTION
        return self._apply_confirmation(pending[0], confirmed, now, tape)

    def _apply_confirmation(
        self, request: ConfirmationRequest, confirmed: bool | None, now: datetime, tape: "_Tape"
    ) -> AgentResult:
        try:
            token = request.resolve(confirmed, now)
        except TimeoutError:
            return tape.done(
                AgentResult("refused", "That request timed out, so I did nothing. Tell me again if you still want it.")
            )
        except ValueError as exc:
            return tape.done(AgentResult("error", str(exc)))

        if token is None:
            tape.think("declined -- no destructive action taken")
            return tape.done(
                AgentResult("refused", "Left it unchanged.", decision=request.decision, confirmation=request)
            )

        try:
            self.engine.apply_destructive(token)
        except RateLimitError:
            return tape.done(AgentResult("failure", "The calendar is rate-limited; I queued the change."))

        action = request.decision.action
        tape.tool("calendar.%s" % action.value.lower(), {"event_id": request.decision.target_event_id}, "ok")
        tape.act(action, request.decision.target_event_id, request.request_id)
        verb = "Deleted" if action is ActionType.DELETE else "Updated"
        return tape.done(
            AgentResult(
                "success",
                "%s the calendar event and its alarm." % verb,
                decision=request.decision,
                confirmation=request,
            )
        )

    # -- NFR4 activation window --------------------------------------------

    def _outside_window(self, now: datetime, tape: "_Tape") -> AgentResult | None:
        """Refuse if the user is outside their window; otherwise release queues.

        Returns a finished refusal, or None to mean "carry on". With
        ``activation=None`` it is a single None-check and the agent behaves
        exactly as it did before Phase 5.
        """
        if self.activation is None:
            return None
        if self.activation.contains(now):
            self._release_queued(now, tape)
            return None
        tape.think("outside the active window %s" % self.activation.describe())
        return tape.done(
            AgentResult(
                "refused",
                "I am only active between %s. I have not changed anything -- tell me again after %s."
                % (self.activation.describe(), self.activation.next_start(now).strftime("%H:%M")),
            )
        )

    def _release_queued(self, now: datetime, tape: "_Tape") -> None:
        """Back inside the window, anything parked by UC3 E2 becomes askable again.

        Without this a queued request is invisible to ``pending()`` for good, and
        a later "yes" would find nothing to attach itself to.
        """
        for request in release_queued(self.confirmations, self.activation, now):
            tape.think("released queued confirmation %s for delivery (UC3 E2)" % request.request_id)

    # -- shared path -------------------------------------------------------

    def _reconcile_from_frame(self, frame: Frame, source: Source, now: datetime, tape: "_Tape") -> AgentResult:
        slots = {s.name: s.value for s in frame.slots}
        status = _INTENT_STATUS[frame.intent]
        events = self.calendar.list_events()

        when: datetime | None = None
        title = slots.get("relative_ref") or _title_from(slots)

        resolution = resolve_datetime(slots.get("date"), slots.get("time"), now)
        if resolution.ok:
            when = resolution.value
            if resolution.note:
                tape.think(resolution.note)
        elif "relative_ref" in slots or "booking_ref" in slots:
            # "Cancel my Chennai flight" carries no date -- the calendar supplies it.
            candidates = self._candidates_for(slots, events)
            if len(candidates) == 1:
                when = candidates[0].start_time
                tape.think("resolved %r to event %s at %s" % (title, candidates[0].event_id, when))
            elif len(candidates) > 1:
                tape.think("%d events match %r -- asking rather than guessing" % (len(candidates), title))
                return tape.done(
                    AgentResult(
                        "refused",
                        "More than one booking matches that: %s. Which one?"
                        % ", ".join("%s on %s" % (c.title, c.start_time.strftime("%d %b at %H:%M")) for c in candidates),
                        needs_clarification=True,
                    )
                )

        if when is None:
            # UC1 E2 -- ask, never guess. A wrong date is a missed flight.
            tape.think(resolution.note or "no date available")
            return tape.done(
                AgentResult(
                    "refused",
                    "I need a date for that. When is it?",
                    needs_clarification=True,
                )
            )

        if not title:
            return tape.done(
                AgentResult("refused", "I could not tell which booking you meant.", needs_clarification=True)
            )

        booking = _build_booking(
            booking_type=_type_from(slots),
            status=status,
            when=when,
            title=title,
            source=source,
            booking_ref=slots.get("booking_ref"),
            destination=slots.get("destination"),
            location=slots.get("location"),
        )
        return self._decide_and_act(booking, now, tape)

    def _candidates_for(self, slots: dict, events: list[CalendarEvent]) -> list[CalendarEvent]:
        ref = slots.get("booking_ref")
        if ref:
            exact = [e for e in events if e.booking_ref and e.booking_ref.upper() == ref.upper()]
            if exact:
                return exact
        phrase = slots.get("relative_ref")
        return find_by_phrase(events, phrase) if phrase else []

    def _decide_and_act(self, booking: BookingEvent, now: datetime, tape: "_Tape") -> AgentResult:
        events = self.calendar.list_events()
        tape.tool("calendar.list_events", {}, len(events))
        decision = decide(events, booking)
        tape.think("%s: %s" % (decision.reason.value, decision.note or decision.describe()))

        if decision.reason is Reason.AMBIGUOUS_MATCH:
            return tape.done(
                AgentResult(
                    "refused",
                    "More than one calendar event could be that booking, so I have not touched any of them.",
                    decision=decision,
                    needs_clarification=True,
                )
            )

        if not decision.requires_confirmation:
            try:
                created = self.engine.apply(decision)
            except RateLimitError:
                return tape.done(AgentResult("failure", "The calendar is rate-limited; I queued the change."))
            if decision.action is ActionType.CREATE and created is not None:
                tape.tool("calendar.create_event", {"title": booking.title}, created.event_id)
                tape.act(ActionType.CREATE, created.event_id, None)
                return tape.done(
                    AgentResult(
                        "success",
                        "Added %s on %s to your calendar." % (booking.title, booking.date_time.strftime("%d %b at %H:%M")),
                        decision=decision,
                    )
                )
            tape.act(ActionType.NOOP, decision.target_event_id, None)
            return tape.done(AgentResult("success", "Your calendar already reflects that.", decision=decision))

        # Destructive. NFR2: reuse an open request rather than asking twice --
        # including one parked outside the window, which `find_pending_for`
        # cannot see because a queued request is no longer PENDING.
        existing = find_open_confirmation(self.confirmations, booking.booking_ref, decision.target_event_id, now)
        if existing is not None:
            if existing.status is ConfirmationStatus.QUEUED:
                # Saying "I already asked" would be false: the user was never
                # asked, because they were unreachable when it came up.
                message = "I already have that waiting to ask you: %s" % existing.prompt
            else:
                message = "I already asked about that: %s" % existing.prompt
            return tape.done(
                AgentResult("awaiting_confirmation", message, decision=decision, confirmation=existing)
            )

        request = self.confirmations.open(decision, now)

        # UC3 E2. The user cannot be reached, so the prompt is not delivered --
        # and the change is neither applied nor dropped. Both of those would be
        # defects: applying it is an FR5 violation, dropping it leaves a
        # cancelled booking on the calendar.
        if queue_outside_window(request, self.activation, now):
            opens = self.activation.next_start(now)
            tape.think(
                "user is outside their active window (%s); queued %s until %s (UC3 E2)"
                % (self.activation.describe(), request.request_id, opens.isoformat())
            )
            return tape.done(
                AgentResult(
                    "awaiting_confirmation",
                    "This needs your confirmation, but it is outside your active hours. "
                    "I have changed nothing and will ask you after %s: %s"
                    % (opens.strftime("%H:%M"), request.prompt),
                    decision=decision,
                    confirmation=request,
                )
            )

        tape.tool("user.prompt_confirmation", {"request_id": request.request_id}, request.prompt)
        return tape.done(
            AgentResult("awaiting_confirmation", request.prompt, decision=decision, confirmation=request)
        )


# -- helpers --------------------------------------------------------------


def _summarise(events: list[CalendarEvent]) -> str:
    return "; ".join("%s on %s" % (e.title, e.start_time.strftime("%d %b at %H:%M")) for e in events)


def _type_from(slots: dict) -> BookingType:
    raw = (slots.get("booking_type") or "").lower()
    for booking_type in BookingType:
        if booking_type.value in raw:
            return booking_type
    if "appointment" in raw or slots.get("provider"):
        return BookingType.APPOINTMENT
    return BookingType.FLIGHT


def _title_from(slots: dict) -> str | None:
    parts = [slots.get("booking_type"), "to" if slots.get("destination") else None, slots.get("destination")]
    title = " ".join(p for p in parts if p).strip()
    return title or slots.get("provider")


def _build_booking(
    booking_type: BookingType,
    status: BookingStatus,
    when: datetime,
    title: str,
    source: Source,
    booking_ref: str | None = None,
    destination: str | None = None,
    location: str | None = None,
) -> BookingEvent:
    cls = _BOOKING_CLASSES[booking_type]
    return cls(
        booking_id=str(uuid.uuid4())[:8],
        booking_type=booking_type,
        date_time=when,
        status=status,
        source=source,
        title=title,
        booking_ref=booking_ref,
        destination=destination,
        location=location,
    )


@dataclass
class _Tape:
    """Records one interaction in the frozen trajectory schema (FR7)."""

    source: str
    text: str
    now: datetime
    model: str
    reasoning: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)

    def think(self, note: str) -> None:
        self.reasoning.append({"step": len(self.reasoning) + 1, "text": note, "ts": self.now.isoformat()})

    def tool(self, name: str, args: dict, result) -> None:
        self.tool_calls.append(
            {"name": name, "args": args, "result": result, "ok": True, "error": None, "ts": self.now.isoformat()}
        )

    def act(self, action: ActionType, target: str | None, confirmation_id: str | None) -> None:
        destructive = action in (ActionType.DELETE, ActionType.UPDATE)
        self.actions.append(
            {
                "type": action.value,
                "target": target,
                "destructive": destructive,
                "confirmed_by_user": bool(confirmation_id) if destructive else None,
                "confirmation_id": confirmation_id,
            }
        )

    def done(self, result: AgentResult) -> AgentResult:
        result.trajectory = {
            "trajectory_id": str(uuid.uuid4()),
            "project_id": "booking-sync",
            "archetype": "tool_using_agent",
            "agent": {"model": self.model, "prompt_version": None, "commit": None},
            "task": {"source": self.source, "input": self.text, "use_case": None},
            "reasoning": self.reasoning or None,
            "tool_calls": self.tool_calls,
            "actions": self.actions,
            "outcome": {"status": result.status, "user_visible": result.message},
            "timing": {"started": self.now.isoformat(), "ended": self.now.isoformat(), "total_ms": None},
            "scenario_id": None,
            "scores": None,
            "schema_version": "1.0",
        }
        return result
