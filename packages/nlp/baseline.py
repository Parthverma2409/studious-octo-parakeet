"""Rule-based extractor: the measurable floor, and the backstop when the LLM fails.

Two jobs:

1. **A floor.** "The LLM scores 0.81 frame accuracy" means nothing until you know
   what keyword matching scores. If a 7B model cannot beat regexes on this task,
   that is the finding.
2. **A backstop.** The plan's risk table calls for rules to cover a model that
   returns unusable output; `FallbackExtractor` wires this in behind it.

It deliberately does NOT try to be clever. Where it is unsure it emits no slot and
falls back to out_of_scope -- under-extraction is recoverable, a confident wrong
answer is not.
"""

from __future__ import annotations

import re
from pathlib import Path

from .generate import load_lexicon
from .schema import Frame, Intent, Slot, SLOT_NAMES

# Order matters: the first pattern to match wins, so the destructive and
# response intents are tested before the broader booking verbs.
_INTENT_RULES: list[tuple[Intent, re.Pattern]] = [
    # Hesitation and self-correction are never consent (UC3 open issue). Checked
    # first so "no wait, cancel it" can never be read as a cancel or a confirm.
    (Intent.DECLINE_ACTION, re.compile(r"\b(no wait|hold on|actually never mind|never mind|leave it|leave the|stop|keep it|don'?t)\b", re.I)),
    (Intent.DECLINE_ACTION, re.compile(r"\b(no|nope|that'?s not right)\b", re.I)),
    (Intent.CONFIRM_ACTION, re.compile(r"\b(yes|yeah|yep|confirm|go ahead|do it|do that|that'?s correct|i'?m sure)\b", re.I)),
    (Intent.RESCHEDULE_BOOKING, re.compile(r"\b(reschedule|prepone|postpone|move|shift|push|change)\b", re.I)),
    # "make my flight Friday instead of the current one" -- a reschedule carrying
    # no reschedule verb at all.
    (Intent.RESCHEDULE_BOOKING, re.compile(r"\bmake\b.*\binstead\b", re.I)),
    (Intent.CANCEL_BOOKING, re.compile(r"\b(cancel|scrap|call off|drop|remove)\b", re.I)),
    (Intent.CANCEL_BOOKING, re.compile(r"\bdon'?t need\b.*\banymore\b", re.I)),
    (Intent.QUERY_STATUS, re.compile(r"\b(when|what time|status|still confirmed|has .* been|do i have|tell me about|check my)\b", re.I)),
    (Intent.CREATE_BOOKING, re.compile(r"\b(book|reserve|schedule|set up|get me on|i want to fly|add a)\b", re.I)),
]

# Out-of-scope wins outright: these say the utterance is not about a booking at
# all, however booking-shaped the rest of the sentence looks.
_OUT_OF_SCOPE = re.compile(
    r"\b(weather|how much|cost|how far|timer|best time to visit|tell me a joke|play some|airport busy|what time is it in|remind me to)\b",
    re.I,
)

_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_WEEKDAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"

# Dates and times are recognised by SHAPE, not by membership of the lexicon.
# A list-driven matcher only ever finds the dates it was shown, which is useless
# the moment a real user says a month the list happens not to contain.
# Longest-first: "next Friday" must beat "Friday".
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:%s)\b" % _MONTHS, re.I),
    re.compile(r"\b(?:%s)\s+\d{1,2}(?:st|nd|rd|th)?\b" % _MONTHS, re.I),
    re.compile(r"\b(?:the\s+)?day after tomorrow\b", re.I),
    re.compile(r"\b(?:next|this|coming)\s+(?:%s|weekend)\b" % _WEEKDAYS, re.I),
    re.compile(r"\b(?:%s)\b" % _WEEKDAYS, re.I),
    re.compile(r"\b(?:today|tomorrow)\b", re.I),
    re.compile(r"\bthe\s+\d{1,2}(?:st|nd|rd|th)\b", re.I),
]

_TIME_PATTERNS = [
    re.compile(r"\b\d{1,2}\s+in\s+the\s+(?:morning|afternoon|evening|night)\b", re.I),
    re.compile(r"\b(?:half|quarter)\s+(?:past|to)\s+\w+\b", re.I),
    re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:am|pm)\b", re.I),
    re.compile(r"\b\d{1,2}[:.]\d{2}\b"),
    re.compile(r"\b\d{1,2}\s*(?:am|pm)\b", re.I),
    re.compile(r"\b(?:early morning|late evening|noon|midnight|morning|afternoon|evening)\b", re.I),
]

_BOOKING_REF = re.compile(r"\b[A-Z0-9]{2,6}-[A-Z0-9]{2,6}\b")
_PARTY_SIZE = re.compile(r"\b(?:for\s+)?(one|two|three|four|five|a couple of|\d+)\s+people\b", re.I)


class BaselineExtractor:
    name = "rule-based-v1"

    def __init__(self, lexicon_path: Path | None = None):
        lex = load_lexicon(lexicon_path)
        self.cities = sorted(lex["city"], key=len, reverse=True)
        self.booking_types = sorted(lex["booking_type"], key=len, reverse=True)
        self.dates = sorted(lex["date"], key=len, reverse=True)
        self.times = sorted(lex["time"], key=len, reverse=True)
        self.providers = sorted(lex["provider"], key=len, reverse=True)
        self.relative_refs = sorted(lex["relative_ref"], key=len, reverse=True)

    # -- intent ----------------------------------------------------------

    def classify(self, text: str) -> Intent:
        if _OUT_OF_SCOPE.search(text):
            return Intent.OUT_OF_SCOPE
        for intent, pattern in _INTENT_RULES:
            if pattern.search(text):
                return intent
        return Intent.OUT_OF_SCOPE

    # -- slots -----------------------------------------------------------

    def _claim(self, text: str, taken: list[range], name: str, value: str, start: int) -> Slot | None:
        span = range(start, start + len(value))
        if any(max(span.start, t.start) < min(span.stop, t.stop) for t in taken):
            return None  # already covered by a longer, earlier match
        taken.append(span)
        return Slot(name=name, start=span.start, end=span.stop, value=text[span.start : span.stop])

    def _find_all(self, text: str, name: str, candidates: list[str], taken: list[range]) -> list[Slot]:
        found = []
        lowered = text.lower()
        for candidate in candidates:
            idx = lowered.find(candidate.lower())
            while idx != -1:
                slot = self._claim(text, taken, name, candidate, idx)
                if slot:
                    found.append(slot)
                idx = lowered.find(candidate.lower(), idx + 1)
        return found

    def _find_patterns(self, text: str, name: str, patterns: list[re.Pattern], taken: list[range]) -> list[Slot]:
        found = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                slot = self._claim(text, taken, name, match.group(0), match.start())
                if slot:
                    found.append(slot)
        return found

    def extract_slots(self, text: str) -> tuple[Slot, ...]:
        taken: list[range] = []
        slots: list[Slot] = []

        # Longest and least ambiguous first, so a shorter pattern cannot steal a
        # span from a longer one ("my Chennai flight" before "Chennai").
        for match in _BOOKING_REF.finditer(text):
            slot = self._claim(text, taken, "booking_ref", match.group(0), match.start())
            if slot:
                slots.append(slot)

        slots += self._find_all(text, "relative_ref", self.relative_refs, taken)
        slots += self._find_all(text, "provider", self.providers, taken)
        slots += self._find_patterns(text, "date", _DATE_PATTERNS, taken)
        slots += self._find_patterns(text, "time", _TIME_PATTERNS, taken)
        # Lexicon sweep afterwards, for phrasings the patterns do not cover.
        slots += self._find_all(text, "date", self.dates, taken)
        slots += self._find_all(text, "time", self.times, taken)

        for match in _PARTY_SIZE.finditer(text):
            value = match.group(1)
            slot = self._claim(text, taken, "party_size", value, match.start(1))
            if slot:
                slots.append(slot)

        slots += self._find_all(text, "booking_type", self.booking_types, taken)

        # Cities are only origin/destination when a preposition says so. An
        # unprefixed city is more often part of a relative reference.
        lowered = text.lower()
        for city in self.cities:
            idx = lowered.find(city.lower())
            while idx != -1:
                prefix = lowered[max(0, idx - 6) : idx]
                name = "origin" if re.search(r"\bfrom\s+$", prefix) else ("destination" if re.search(r"\b(to|in)\s+$", prefix) else None)
                if name:
                    slot = self._claim(text, taken, name, city, idx)
                    if slot:
                        slots.append(slot)
                idx = lowered.find(city.lower(), idx + 1)

        return tuple(sorted(s for s in slots if s.name in SLOT_NAMES))

    # -- port ------------------------------------------------------------

    def extract(self, text: str) -> Frame:
        intent = self.classify(text)
        if intent in (Intent.CONFIRM_ACTION, Intent.DECLINE_ACTION, Intent.OUT_OF_SCOPE):
            # A yes/no answer carries no booking slots, and labelling slots on an
            # out-of-scope utterance is how a non-request becomes an action.
            return Frame(intent=intent, slots=())
        return Frame(intent=intent, slots=self.extract_slots(text))
