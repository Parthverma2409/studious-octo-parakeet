"""Metrics behave as documented -- checked against hand-computed cases.

A metrics bug is the worst kind of bug in an evaluation project: every number in
the final report inherits it, and nothing else in the system misbehaves.
"""

from __future__ import annotations

from packages.nlp.metrics import evaluate, safety_critical_confusions
from packages.nlp.schema import Frame, Intent, Slot, Utterance


def _gold(text, intent, slots=()):
    return Utterance(text=text, intent=intent, slots=tuple(slots))


def test_perfect_prediction_scores_one() -> None:
    gold = [_gold("cancel my flight", Intent.CANCEL_BOOKING, [Slot("relative_ref", 7, 16, "my flight")])]
    pred = [Frame(intent=Intent.CANCEL_BOOKING, slots=(Slot("relative_ref", 7, 16, "my flight"),))]
    report = evaluate(gold, pred)
    assert report.intent_accuracy == 1.0
    assert report.slot_micro.f1 == 1.0
    assert report.frame_accuracy == 1.0


def test_right_intent_wrong_slot_still_fails_the_frame() -> None:
    """The whole reason frame accuracy is the headline metric: this case produces
    a confidently wrong calendar entry while intent accuracy reads 1.0."""
    gold = [_gold("cancel my flight on Monday", Intent.CANCEL_BOOKING, [Slot("date", 20, 26, "Monday")])]
    pred = [Frame(intent=Intent.CANCEL_BOOKING, slots=(Slot("date", 0, 6, "cancel"),))]
    report = evaluate(gold, pred)
    assert report.intent_accuracy == 1.0
    assert report.frame_accuracy == 0.0


def test_missing_slot_costs_recall_not_precision() -> None:
    gold = [
        _gold(
            "book a flight on Monday",
            Intent.CREATE_BOOKING,
            [Slot("booking_type", 7, 13, "flight"), Slot("date", 17, 23, "Monday")],
        )
    ]
    pred = [Frame(intent=Intent.CREATE_BOOKING, slots=(Slot("booking_type", 7, 13, "flight"),))]
    report = evaluate(gold, pred)
    assert report.slot_micro.precision == 1.0
    assert report.slot_micro.recall == 0.5


def test_spurious_slot_costs_precision_not_recall() -> None:
    gold = [_gold("book a flight", Intent.CREATE_BOOKING, [Slot("booking_type", 7, 13, "flight")])]
    pred = [
        Frame(
            intent=Intent.CREATE_BOOKING,
            slots=(Slot("booking_type", 7, 13, "flight"), Slot("date", 0, 4, "book")),
        )
    ]
    report = evaluate(gold, pred)
    assert report.slot_micro.recall == 1.0
    assert report.slot_micro.precision == 0.5


def test_same_span_wrong_slot_name_is_not_a_match() -> None:
    gold = [_gold("fly to Chennai", Intent.CREATE_BOOKING, [Slot("destination", 7, 14, "Chennai")])]
    pred = [Frame(intent=Intent.CREATE_BOOKING, slots=(Slot("origin", 7, 14, "Chennai"),))]
    assert evaluate(gold, pred).slot_micro.f1 == 0.0


def test_confusions_are_reported() -> None:
    gold = [_gold("no wait", Intent.DECLINE_ACTION), _gold("no wait", Intent.DECLINE_ACTION)]
    pred = [Frame(intent=Intent.CONFIRM_ACTION), Frame(intent=Intent.CONFIRM_ACTION)]
    report = evaluate(gold, pred)
    assert ("decline_action", "confirm_action", 2) in report.intent_confusions


def test_parse_failures_are_counted() -> None:
    gold = [_gold("something", Intent.CREATE_BOOKING)]
    pred = [Frame(intent=Intent.OUT_OF_SCOPE, parse_failed=True)]
    assert evaluate(gold, pred).parse_failures == 1


def test_decline_read_as_confirm_is_safety_critical() -> None:
    """This single confusion manufactures the consent FR5 depends on."""
    gold = [_gold("no wait, don't cancel it", Intent.DECLINE_ACTION)]
    pred = [Frame(intent=Intent.CONFIRM_ACTION)]
    assert safety_critical_confusions(gold, pred) == 1


def test_out_of_scope_read_as_cancel_is_safety_critical() -> None:
    gold = [_gold("what's the weather in Chennai", Intent.OUT_OF_SCOPE)]
    pred = [Frame(intent=Intent.CANCEL_BOOKING)]
    assert safety_critical_confusions(gold, pred) == 1


def test_harmless_confusion_is_not_safety_critical() -> None:
    """Under-reacting is not the failure mode we gate on."""
    gold = [_gold("book a flight", Intent.CREATE_BOOKING)]
    pred = [Frame(intent=Intent.OUT_OF_SCOPE)]
    assert safety_critical_confusions(gold, pred) == 0
