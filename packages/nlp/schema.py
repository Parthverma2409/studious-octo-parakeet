"""The NLU frame -- the contract between the NLP layer and the Sync Agent.

A frame is what FR1/FR6 actually need: one intent plus the slots that let the
ReconciliationEngine build a BookingEvent. Keeping it a plain dataclass (rather
than raw model output) means the LLM extractor, the rule-based baseline and the
gold data are all comparable by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    CREATE_BOOKING = "create_booking"
    RESCHEDULE_BOOKING = "reschedule_booking"
    CANCEL_BOOKING = "cancel_booking"
    QUERY_STATUS = "query_status"
    CONFIRM_ACTION = "confirm_action"
    DECLINE_ACTION = "decline_action"
    OUT_OF_SCOPE = "out_of_scope"


#: The two intents that answer a ConfirmationRequest (UC3). Kept explicit so a
#: caller can never treat "anything that isn't a decline" as consent.
RESPONSE_INTENTS = frozenset({Intent.CONFIRM_ACTION, Intent.DECLINE_ACTION})

SLOT_NAMES = (
    "booking_type",
    "origin",
    "destination",
    "date",
    "time",
    "provider",
    "booking_ref",
    "party_size",
    "relative_ref",
)


@dataclass(frozen=True, order=True)
class Slot:
    name: str
    start: int
    end: int
    value: str

    def key(self) -> tuple[str, int, int]:
        """Span identity, used for exact-match scoring (seqeval strict mode)."""
        return (self.name, self.start, self.end)

    def to_dict(self) -> dict:
        return {"name": self.name, "start": self.start, "end": self.end, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Slot":
        return cls(d["name"], int(d["start"]), int(d["end"]), d["value"])


@dataclass(frozen=True)
class Utterance:
    """One labelled example."""

    text: str
    intent: Intent
    slots: tuple[Slot, ...] = ()
    template_id: str | None = None
    source: str = "template"

    def slot_keys(self) -> set[tuple[str, int, int]]:
        return {s.key() for s in self.slots}

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "intent": self.intent.value,
            "slots": [s.to_dict() for s in self.slots],
            "template_id": self.template_id,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(
            text=d["text"],
            intent=Intent(d["intent"]),
            slots=tuple(Slot.from_dict(s) for s in d.get("slots", [])),
            template_id=d.get("template_id"),
            source=d.get("source", "template"),
        )

    def validate(self) -> list[str]:
        """Offsets must actually point at the value they claim.

        Generated data drifts silently otherwise: a wrong offset trains and scores
        against text that is not there, and nothing else in the pipeline notices.
        """
        problems = []
        for s in self.slots:
            if s.name not in SLOT_NAMES:
                problems.append("unknown slot %r" % s.name)
            if not (0 <= s.start < s.end <= len(self.text)):
                problems.append("slot %s has out-of-range span %d:%d" % (s.name, s.start, s.end))
            elif self.text[s.start : s.end] != s.value:
                problems.append(
                    "slot %s span %d:%d is %r, not %r"
                    % (s.name, s.start, s.end, self.text[s.start : s.end], s.value)
                )
        spans = sorted((s.start, s.end) for s in self.slots)
        for (_, end), (start, _) in zip(spans, spans[1:]):
            if start < end:
                problems.append("overlapping slot spans")
                break
        return problems


@dataclass
class Frame:
    """What an extractor returns. `slots` may be partial; `raw` keeps the model
    output for the trajectory, so a parse failure is inspectable later."""

    intent: Intent
    slots: tuple[Slot, ...] = ()
    raw: str | None = None
    parse_failed: bool = False
    notes: list[str] = field(default_factory=list)
