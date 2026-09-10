"""Expand templates x lexicon into labelled utterances with exact slot offsets.

Offsets are computed while the string is built, never searched for afterwards.
Searching is where generated NLU data goes quietly wrong: a value that appears
twice in one utterance ("Chennai to Chennai") gets the first match labelled and
the second silently unlabelled.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import yaml

from .schema import Intent, Slot, Utterance

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "nlu"
#: `{slot}` samples from the lexicon; `{slot=literal}` pins a fixed value so a
#: word that is genuinely part of the sentence still gets labelled. Without the
#: second form, "schedule an appointment with Dr Rao" leaves "appointment"
#: unlabelled and every extractor is marked wrong for correctly finding it.
PLACEHOLDER = re.compile(r"\{([a-z_]+)(?:=([^}]+))?\}")

#: Slots that draw from the shared `city` list rather than a list of their own.
_CITY_SLOTS = {"origin", "destination"}


def load_lexicon(path: Path | None = None) -> dict[str, list[str]]:
    lexicon = yaml.safe_load((path or DATA_DIR / "lexicon.yaml").read_text(encoding="utf-8"))
    # YAML 1.1 silently reads a bare `18:40` as the integer 1120 and `09` as 9.
    # Caught here rather than deep in span construction, where the symptom is
    # unrecognisable and the corrupted value has already reached the dataset.
    for slot, values in lexicon.items():
        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    "lexicon %s contains a non-string value %r (%s) -- quote it in lexicon.yaml"
                    % (slot, value, type(value).__name__)
                )
    return lexicon


def load_templates(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or DATA_DIR / "templates.yaml").read_text(encoding="utf-8"))


def _values_for(slot: str, lexicon: dict[str, list[str]]) -> list[str]:
    if slot in _CITY_SLOTS:
        return lexicon["city"]
    if slot not in lexicon:
        raise KeyError("no lexicon entry for slot %r" % slot)
    return lexicon[slot]


def _render(template_text: str, choice: dict[str, str]) -> tuple[str, tuple[Slot, ...]]:
    """Build the utterance and its slot spans in one pass."""
    out: list[str] = []
    slots: list[Slot] = []
    cursor = 0
    pos = 0
    for match in PLACEHOLDER.finditer(template_text):
        literal = template_text[pos : match.start()]
        out.append(literal)
        cursor += len(literal)

        name, fixed = match.group(1), match.group(2)
        value = fixed if fixed is not None else choice[name]
        slots.append(Slot(name=name, start=cursor, end=cursor + len(value), value=value))
        out.append(value)
        cursor += len(value)
        pos = match.end()

    out.append(template_text[pos:])
    return "".join(out), tuple(slots)


def expand_template(
    template: dict,
    lexicon: dict[str, list[str]],
    max_variants: int,
    rng: random.Random,
) -> list[Utterance]:
    """Generate up to `max_variants` distinct fills of one template.

    A template with no placeholders yields exactly one utterance -- emitting
    eighteen identical copies of "confirm" would inflate the dataset without
    adding a single bit of information.
    """
    # Only free placeholders need sampling; fixed ones are already decided.
    names = [n for n, fixed in PLACEHOLDER.findall(template["text"]) if not fixed]
    if not names:
        return [
            Utterance(
                text=template["text"],
                intent=Intent(template["intent"]),
                slots=(),
                template_id=template["id"],
            )
        ]

    seen: set[str] = set()
    produced: list[Utterance] = []
    attempts = 0
    ceiling = max_variants * 8 + 32

    while len(produced) < max_variants and attempts < ceiling:
        attempts += 1
        choice = {}
        for name in names:
            choice[name] = rng.choice(_values_for(name, lexicon))
        # "a flight from Chennai to Chennai" is not a sentence anyone says.
        if "origin" in choice and "destination" in choice and choice["origin"] == choice["destination"]:
            continue
        text, slots = _render(template["text"], choice)
        if text in seen:
            continue
        seen.add(text)
        intent = Intent(template["intent"])
        # An out-of-scope utterance produces no booking, so it has no booking
        # slots -- "what's the weather in Chennai" has a city in it, but not a
        # destination. Labelling one would teach the extractor to find booking
        # fields in sentences that are not bookings.
        if intent is Intent.OUT_OF_SCOPE:
            slots = ()
        produced.append(Utterance(text=text, intent=intent, slots=slots, template_id=template["id"]))

    return produced


def generate(
    templates: list[dict] | None = None,
    lexicon: dict[str, list[str]] | None = None,
    variants_per_template: int = 18,
    seed: int = 20260907,
) -> list[Utterance]:
    """Deterministic for a given seed, so regenerating never silently reshuffles
    what is already split into train/dev/test."""
    templates = templates if templates is not None else load_templates()
    lexicon = lexicon if lexicon is not None else load_lexicon()

    ids = [t["id"] for t in templates]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError("duplicate template ids: %s" % sorted(duplicates))

    utterances: list[Utterance] = []
    for template in templates:
        # Seed per template so adding a template never changes the others' output.
        rng = random.Random("%s:%d" % (template["id"], seed))
        utterances.extend(expand_template(template, lexicon, variants_per_template, rng))

    problems = [(u.text, p) for u in utterances for p in u.validate()]
    if problems:
        raise ValueError("generated invalid utterances: %s" % problems[:5])

    return utterances
