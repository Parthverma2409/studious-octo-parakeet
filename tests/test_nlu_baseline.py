"""Rule-based extractor, and the parsing of model output.

The behaviours pinned here are the ones with safety consequences. Everything else
about the baseline is allowed to be mediocre -- it is a floor, not a product.
"""

from __future__ import annotations

import pytest

from packages.nlp.baseline import BaselineExtractor
from packages.nlp.extractor import locate_slots, parse_frame
from packages.nlp.schema import Intent

EXTRACTOR = BaselineExtractor()


@pytest.mark.parametrize(
    "text",
    [
        "no don't do that",
        "no wait, don't cancel it",
        "actually never mind",
        "hold on, leave the booking alone",
        "stop, keep it",
        "cancel that, I mean don't cancel the flight",
    ],
)
def test_a_decline_is_never_read_as_a_confirmation(text: str) -> None:
    """The worst error this layer can make: manufacturing the consent FR5 needs."""
    assert EXTRACTOR.classify(text) is not Intent.CONFIRM_ACTION


@pytest.mark.parametrize(
    "text",
    ["cancel that, I mean don't cancel the flight", "no wait, don't cancel it"],
)
def test_self_correction_does_not_become_a_cancellation(text: str) -> None:
    """"Cancel" appears in both, but neither utterance is a cancel instruction."""
    assert EXTRACTOR.classify(text) is Intent.DECLINE_ACTION


@pytest.mark.parametrize(
    "text",
    [
        "what's the weather in Chennai",
        "how much does a flight to Delhi cost",
        "how far is Goa from Pune",
        "tell me a joke",
        "remind me to call Dr Rao",
    ],
)
def test_booking_adjacent_chatter_stays_out_of_scope(text: str) -> None:
    assert EXTRACTOR.classify(text) is Intent.OUT_OF_SCOPE


def test_out_of_scope_yields_no_slots() -> None:
    """Extracting a destination from small talk is how a non-request becomes an action."""
    assert EXTRACTOR.extract("what's the weather in Chennai").slots == ()


def test_confirmation_carries_no_booking_slots() -> None:
    assert EXTRACTOR.extract("yes go ahead").slots == ()


def test_booking_reference_is_extracted() -> None:
    frame = EXTRACTOR.extract("cancel booking AI440-XY12")
    assert frame.intent is Intent.CANCEL_BOOKING
    assert [(s.name, s.value) for s in frame.slots] == [("booking_ref", "AI440-XY12")]


def test_origin_and_destination_are_distinguished_by_preposition() -> None:
    frame = EXTRACTOR.extract("book a flight from Delhi to Chennai on Monday")
    found = {s.name: s.value for s in frame.slots}
    assert found["origin"] == "Delhi"
    assert found["destination"] == "Chennai"


def test_bare_city_is_not_claimed_as_a_destination() -> None:
    """Without a preposition a city is usually part of a relative reference."""
    frame = EXTRACTOR.extract("cancel my Chennai flight")
    assert "destination" not in {s.name for s in frame.slots}


def test_extracted_spans_are_valid() -> None:
    for text in [
        "book a flight from Delhi to Chennai on Monday at noon",
        "shift the morning train to half past eight",
        "cancel booking IR-55512",
    ]:
        frame = EXTRACTOR.extract(text)
        for slot in frame.slots:
            assert text[slot.start : slot.end] == slot.value


# -- model-output parsing -------------------------------------------------


def test_parse_frame_reads_well_formed_output() -> None:
    raw = '{"intent": "cancel_booking", "slots": [{"name": "relative_ref", "value": "my flight"}]}'
    frame = parse_frame("cancel my flight", raw)
    assert frame.intent is Intent.CANCEL_BOOKING
    assert not frame.parse_failed
    assert [(s.name, s.start, s.end) for s in frame.slots] == [("relative_ref", 7, 16)]


def test_parse_frame_tolerates_prose_around_the_json() -> None:
    raw = 'Sure! Here you go:\n{"intent": "confirm_action", "slots": []}\nHope that helps.'
    assert parse_frame("yes", raw).intent is Intent.CONFIRM_ACTION


def test_parse_frame_never_raises_on_garbage() -> None:
    for raw in ["", "not json at all", "{broken", '{"intent": "teleport"}']:
        frame = parse_frame("whatever", raw)
        assert frame.parse_failed
        assert frame.intent is Intent.OUT_OF_SCOPE


def test_hallucinated_slot_values_are_dropped() -> None:
    """A value the model invented is not in the utterance, so it has no span."""
    raw = '{"intent": "create_booking", "slots": [{"name": "destination", "value": "Reykjavik"}]}'
    assert parse_frame("book a flight to Chennai", raw).slots == ()


def test_repeated_value_maps_to_distinct_spans() -> None:
    text = "move my Chennai flight from Chennai"
    slots = locate_slots(text, [("destination", "Chennai"), ("origin", "Chennai")])
    assert len({s.start for s in slots}) == 2
