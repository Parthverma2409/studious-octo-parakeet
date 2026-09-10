"""Generation and splitting (D1).

The leakage tests are the important ones. Everything else in the NLU pipeline can
be wrong and you will see it in the numbers; leakage is wrong in a way that makes
the numbers look better.
"""

from __future__ import annotations

import pytest

from packages.nlp.dataset import LeakageError, Splits, assert_no_leakage, split_by_template
from packages.nlp.generate import expand_template, generate, load_lexicon
from packages.nlp.schema import Intent, Slot, Utterance

LEXICON = load_lexicon()


def test_generated_offsets_point_at_their_values() -> None:
    """Every slot span must actually contain the value it claims."""
    for utterance in generate():
        assert utterance.validate() == [], (utterance.text, utterance.validate())


def test_generation_is_deterministic() -> None:
    """Regenerating must not silently reshuffle data already split into train/test."""
    first = [u.text for u in generate()]
    second = [u.text for u in generate()]
    assert first == second


def test_changing_the_seed_changes_the_data() -> None:
    assert [u.text for u in generate(seed=1)] != [u.text for u in generate(seed=2)]


def test_origin_and_destination_are_never_the_same_city() -> None:
    for utterance in generate():
        cities = {s.value for s in utterance.slots if s.name in ("origin", "destination")}
        names = [s.name for s in utterance.slots if s.name in ("origin", "destination")]
        if len(names) == 2:
            assert len(cities) == 2, utterance.text


def test_fixed_placeholder_is_labelled() -> None:
    """`{booking_type=appointment}` labels a literal word that is really a slot."""
    template = {"id": "t1", "intent": "create_booking", "text": "schedule an {booking_type=appointment} with {provider}"}
    rng = __import__("random").Random(0)
    produced = expand_template(template, LEXICON, 3, rng)
    assert produced
    for utterance in produced:
        names = {s.name for s in utterance.slots}
        assert "booking_type" in names
        assert utterance.validate() == []


def test_out_of_scope_utterances_carry_no_slots() -> None:
    """A city in "what's the weather in Chennai" is not a destination."""
    for utterance in generate():
        if utterance.intent is Intent.OUT_OF_SCOPE:
            assert utterance.slots == (), utterance.text


def test_template_without_placeholders_yields_one_row() -> None:
    template = {"id": "c1", "intent": "confirm_action", "text": "confirm"}
    rng = __import__("random").Random(0)
    assert len(expand_template(template, LEXICON, 18, rng)) == 1


def test_non_string_lexicon_value_is_rejected(tmp_path) -> None:
    """YAML 1.1 turns a bare 18:40 into 1120. Fail loudly, not deep in span maths."""
    path = tmp_path / "lex.yaml"
    path.write_text("time:\n  - 18:40\n", encoding="utf-8")
    with pytest.raises(TypeError, match="non-string value"):
        load_lexicon(path)


# -- splitting ------------------------------------------------------------


def test_split_keeps_template_families_intact() -> None:
    splits = split_by_template(generate())
    for name, rows in splits.as_items():
        assert rows, "%s split is empty" % name
    assert_no_leakage(splits)  # redundant with split_by_template, deliberately


def test_every_intent_appears_in_every_split() -> None:
    """Unstratified shuffling can drop confirm/decline out of test entirely -- the
    two intents whose misreading is most costly (UC3)."""
    splits = split_by_template(generate())
    for name, rows in splits.as_items():
        intents = {r.intent for r in rows}
        assert intents == set(Intent), "%s is missing %s" % (name, set(Intent) - intents)


def test_leakage_is_detected_when_a_family_spans_splits() -> None:
    shared = Utterance(text="book a flight", intent=Intent.CREATE_BOOKING, template_id="fam_1")
    other = Utterance(text="book a train", intent=Intent.CREATE_BOOKING, template_id="fam_1")
    with pytest.raises(LeakageError, match="template families"):
        assert_no_leakage(Splits(train=[shared], dev=[], test=[other]))


def test_leakage_is_detected_on_identical_text() -> None:
    a = Utterance(text="cancel my flight", intent=Intent.CANCEL_BOOKING, template_id="fam_1")
    b = Utterance(text="cancel my flight", intent=Intent.CANCEL_BOOKING, template_id="fam_2")
    with pytest.raises(LeakageError, match="identical utterances"):
        assert_no_leakage(Splits(train=[a], dev=[], test=[b]))


def test_splitting_is_deterministic() -> None:
    a = split_by_template(generate())
    b = split_by_template(generate())
    assert [u.text for u in a.test] == [u.text for u in b.test]


def test_slot_offsets_survive_a_jsonl_round_trip(tmp_path) -> None:
    from packages.nlp.dataset import read_jsonl, write_jsonl

    rows = generate()[:25]
    path = tmp_path / "rows.jsonl"
    write_jsonl(rows, path)
    loaded = read_jsonl(path)
    assert [r.to_dict() for r in rows] == [r.to_dict() for r in loaded]
    assert all(r.validate() == [] for r in loaded)


def test_slot_equality_is_span_based() -> None:
    assert Slot("date", 0, 4, "20th").key() == ("date", 0, 4)
