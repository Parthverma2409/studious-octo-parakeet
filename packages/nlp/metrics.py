"""NLU evaluation metrics.

Three numbers, and the third is the one that matters:

  intent accuracy  -- did we understand what the user wanted?
  slot micro-F1    -- exact span match, equivalent to seqeval's strict mode
  frame accuracy   -- intent AND every slot exactly right

An utterance with the right intent and a wrong date produces a confidently wrong
calendar entry while slot F1 still looks respectable. Frame accuracy is the only
one of the three that predicts whether the agent behaves correctly, so it is what
the build is held to.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .schema import Frame, Intent, Utterance


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    support: int

    def as_dict(self) -> dict:
        return {"precision": self.precision, "recall": self.recall, "f1": self.f1, "support": self.support}


def _prf(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return PRF(precision, recall, f1, tp + fn)


@dataclass
class NLUReport:
    n: int
    intent_accuracy: float
    frame_accuracy: float
    slot_micro: PRF
    slot_per_name: dict[str, PRF] = field(default_factory=dict)
    intent_confusions: list[tuple[str, str, int]] = field(default_factory=list)
    parse_failures: int = 0

    def summary(self) -> str:
        lines = [
            "n                 : %d" % self.n,
            "intent accuracy   : %.3f" % self.intent_accuracy,
            "slot micro-F1     : %.3f  (P %.3f / R %.3f)"
            % (self.slot_micro.f1, self.slot_micro.precision, self.slot_micro.recall),
            "frame accuracy    : %.3f   <- the one that matters" % self.frame_accuracy,
        ]
        if self.parse_failures:
            lines.append("parse failures    : %d" % self.parse_failures)
        if self.slot_per_name:
            lines.append("")
            lines.append("per-slot F1:")
            for name, prf in sorted(self.slot_per_name.items(), key=lambda kv: kv[1].f1):
                lines.append("  %-14s %.3f  (n=%d)" % (name, prf.f1, prf.support))
        if self.intent_confusions:
            lines.append("")
            lines.append("top intent confusions (gold -> predicted):")
            for gold, pred, count in self.intent_confusions:
                lines.append("  %-22s -> %-22s %d" % (gold, pred, count))
        return "\n".join(lines)


def evaluate(gold: list[Utterance], predicted: list[Frame]) -> NLUReport:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must be the same length (%d vs %d)" % (len(gold), len(predicted)))

    intent_hits = 0
    frame_hits = 0
    tp = fp = fn = 0
    per_name: Counter[tuple[str, str]] = Counter()
    confusions: Counter[tuple[str, str]] = Counter()
    parse_failures = 0

    for g, p in zip(gold, predicted):
        if p.parse_failed:
            parse_failures += 1

        intent_ok = g.intent == p.intent
        intent_hits += intent_ok
        if not intent_ok:
            confusions[(g.intent.value, p.intent.value)] += 1

        gold_keys = g.slot_keys()
        pred_keys = {s.key() for s in p.slots}

        matched = gold_keys & pred_keys
        tp += len(matched)
        fp += len(pred_keys - gold_keys)
        fn += len(gold_keys - pred_keys)

        for key in matched:
            per_name[(key[0], "tp")] += 1
        for key in pred_keys - gold_keys:
            per_name[(key[0], "fp")] += 1
        for key in gold_keys - pred_keys:
            per_name[(key[0], "fn")] += 1

        if intent_ok and gold_keys == pred_keys:
            frame_hits += 1

    names = {name for name, _ in per_name}
    slot_per_name = {
        name: _prf(per_name[(name, "tp")], per_name[(name, "fp")], per_name[(name, "fn")]) for name in sorted(names)
    }

    n = len(gold)
    return NLUReport(
        n=n,
        intent_accuracy=intent_hits / n if n else 0.0,
        frame_accuracy=frame_hits / n if n else 0.0,
        slot_micro=_prf(tp, fp, fn),
        slot_per_name=slot_per_name,
        intent_confusions=[(g, p, c) for (g, p), c in confusions.most_common(5)],
        parse_failures=parse_failures,
    )


def safety_critical_confusions(gold: list[Utterance], predicted: list[Frame]) -> int:
    """Count the confusions that can cause an unintended destructive action.

    Reading a decline as a confirmation is the worst error this layer can make --
    it manufactures the consent FR5 depends on. Reported separately because it is
    rare enough to vanish inside an aggregate accuracy figure while still being
    the only error that can delete a real booking.
    """
    dangerous_from_oos = (Intent.CANCEL_BOOKING, Intent.CONFIRM_ACTION)
    bad = 0
    for g, p in zip(gold, predicted):
        if g.intent is Intent.DECLINE_ACTION and p.intent is Intent.CONFIRM_ACTION:
            bad += 1
        elif g.intent is Intent.OUT_OF_SCOPE and p.intent in dangerous_from_oos:
            bad += 1
    return bad
