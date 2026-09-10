"""Extractor port, plus the Ollama adapter.

The Sync Agent depends on this protocol, never on a concrete model. That is what
lets the scenario suite and the FR10 sweep swap extractors without touching
reconciliation, and what keeps NFR6 (independently deployable NLP layer) true.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from .schema import Frame, Intent, Slot, SLOT_NAMES

@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, text: str) -> Frame: ...


def locate_slots(text: str, pairs: list[tuple[str, str]]) -> tuple[Slot, ...]:
    """Turn (name, value) pairs into spans, consuming left to right.

    Models return values, not offsets. Each value is located after the previous
    match so that a value occurring twice ("Chennai to Chennai") maps to distinct
    spans instead of both collapsing onto the first occurrence.
    """
    lowered = text.lower()
    slots: list[Slot] = []
    cursor = 0
    for name, value in pairs:
        if not value:
            continue
        idx = lowered.find(value.lower(), cursor)
        if idx == -1:
            idx = lowered.find(value.lower())  # out-of-order model output
        if idx == -1:
            continue  # hallucinated a value that is not in the utterance
        slots.append(Slot(name=name, start=idx, end=idx + len(value), value=text[idx : idx + len(value)]))
        cursor = idx + len(value)
    return tuple(sorted(slots))


def parse_frame(text: str, raw: str) -> Frame:
    """Parse model output into a Frame. Never raises."""
    blob = raw.strip()
    match = re.search(r"\{.*\}", blob, re.DOTALL)
    if not match:
        return Frame(intent=Intent.OUT_OF_SCOPE, raw=raw, parse_failed=True, notes=["no JSON object in output"])
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return Frame(intent=Intent.OUT_OF_SCOPE, raw=raw, parse_failed=True, notes=["invalid JSON: %s" % exc])

    try:
        intent = Intent(payload.get("intent"))
    except ValueError:
        return Frame(
            intent=Intent.OUT_OF_SCOPE,
            raw=raw,
            parse_failed=True,
            notes=["unknown intent %r" % payload.get("intent")],
        )

    pairs: list[tuple[str, str]] = []
    for entry in payload.get("slots") or []:
        if not isinstance(entry, dict):
            continue
        name, value = entry.get("name"), entry.get("value")
        if name in SLOT_NAMES and isinstance(value, str):
            pairs.append((name, value))

    return Frame(intent=intent, slots=locate_slots(text, pairs), raw=raw)


PROMPT = """You extract booking intents from short spoken commands.

Return ONLY a JSON object of the form:
{{"intent": "<intent>", "slots": [{{"name": "<slot>", "value": "<exact substring>"}}]}}

Valid intents: {intents}
Valid slots:   {slots}

Rules:
- Every slot value MUST be an exact substring of the utterance. Never normalise,
  expand or reformat it.
- Emit no slot you are unsure of. A missing slot is recoverable; a wrong one is not.
- "confirm_action" and "decline_action" are answers to a yes/no question the
  assistant asked. If the utterance is hesitant, self-correcting or ambiguous,
  it is NOT a confirmation -- use out_of_scope.
- Use out_of_scope for anything that is not about creating, changing, cancelling
  or asking about a booking.

Utterance: {text}
JSON:"""


class OllamaExtractor:
    """Local open model via Ollama (STRQ2: self-hostable, near-zero cost).

    Not exercised in CI -- no model is installed on the build machine. It is kept
    behind the same protocol as the baseline so swapping it in is a one-line change
    and the metrics are directly comparable.
    """

    def __init__(self, model: str = "qwen2.5:7b-instruct", host: str = "http://localhost:11434", retries: int = 2):
        self.model = model
        self.host = host.rstrip("/")
        self.retries = retries
        self.name = "ollama:%s" % model

    def extract(self, text: str) -> Frame:
        prompt = PROMPT.format(
            intents=", ".join(i.value for i in Intent),
            slots=", ".join(SLOT_NAMES),
            text=text,
        )
        last: Frame | None = None
        for _ in range(self.retries + 1):
            raw = self._call(prompt)
            frame = parse_frame(text, raw)
            if not frame.parse_failed:
                return frame
            last = frame
        return last or Frame(intent=Intent.OUT_OF_SCOPE, parse_failed=True, notes=["no response"])

    def _call(self, prompt: str) -> str:
        import urllib.request

        body = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.host + "/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8")).get("response", "")


class FallbackExtractor:
    """Try the primary extractor; fall back when it fails to parse.

    The plan's risk table calls for a rule-based backstop when a 7B model returns
    unusable output. This makes that explicit rather than leaving the agent to
    handle a failed parse on its own.
    """

    def __init__(self, primary: Extractor, backup: Extractor):
        self.primary = primary
        self.backup = backup
        self.name = "%s+fallback(%s)" % (primary.name, backup.name)

    def extract(self, text: str) -> Frame:
        frame = self.primary.extract(text)
        if not frame.parse_failed:
            return frame
        recovered = self.backup.extract(text)
        recovered.notes = list(recovered.notes) + ["recovered by %s" % self.backup.name]
        return recovered
