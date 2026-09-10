"""Train/dev/test splitting -- by template family, never by row.

This is the single most important file in the NLU pipeline and the easiest thing
to get wrong. Splitting generated data row-wise puts paraphrases of the same seed
on both sides of the split, so the model is tested on restatements of what it
trained on. Reported accuracy rises 10-20 points and nothing in the numbers looks
suspicious -- they just look good.

`assert_no_leakage` is therefore a hard check, not a warning, and it is called by
`split_by_template` itself so it cannot be skipped by forgetting to call it.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .schema import Utterance


@dataclass(frozen=True)
class Splits:
    train: list[Utterance]
    dev: list[Utterance]
    test: list[Utterance]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "dev": len(self.dev), "test": len(self.test)}

    def as_items(self) -> list[tuple[str, list[Utterance]]]:
        return [("train", self.train), ("dev", self.dev), ("test", self.test)]


class LeakageError(AssertionError):
    """Raised when a template family or an exact utterance spans two splits."""


def assert_no_leakage(splits: Splits) -> None:
    families: dict[str, set[str]] = {}
    texts: dict[str, set[str]] = {}
    for name, rows in splits.as_items():
        families[name] = {r.template_id for r in rows if r.template_id}
        texts[name] = {r.text for r in rows}

    names = [n for n, _ in splits.as_items()]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_family = families[a] & families[b]
            if shared_family:
                raise LeakageError(
                    "template families in both %s and %s: %s" % (a, b, sorted(shared_family)[:5])
                )
            shared_text = texts[a] & texts[b]
            if shared_text:
                raise LeakageError("identical utterances in both %s and %s: %s" % (a, b, sorted(shared_text)[:3]))


def split_by_template(
    utterances: list[Utterance],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260907,
) -> Splits:
    """Assign whole template families to splits, stratified by intent.

    Stratifying matters here: with only 8 confirm and 8 decline families, an
    unstratified shuffle can easily put zero confirm families in test -- and
    those are the two intents whose misreading is most costly (UC3).
    """
    if not 0.999 < sum(ratios) < 1.001:
        raise ValueError("ratios must sum to 1")

    by_family: dict[str, list[Utterance]] = defaultdict(list)
    for u in utterances:
        by_family[u.template_id or u.text].append(u)

    families_by_intent: dict[str, list[str]] = defaultdict(list)
    for family, rows in by_family.items():
        families_by_intent[rows[0].intent.value].append(family)

    rng = random.Random(seed)
    buckets: dict[str, list[Utterance]] = {"train": [], "dev": [], "test": []}

    for intent in sorted(families_by_intent):
        families = sorted(families_by_intent[intent])
        rng.shuffle(families)
        n = len(families)
        # Guarantee at least one family in dev and test whenever there are
        # enough to go round, rather than letting rounding empty them.
        n_dev = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        if n_dev + n_test >= n:
            n_dev, n_test = (1, 1) if n >= 3 else (0, 0)

        assignment = (
            [("test", f) for f in families[:n_test]]
            + [("dev", f) for f in families[n_test : n_test + n_dev]]
            + [("train", f) for f in families[n_test + n_dev :]]
        )
        for split_name, family in assignment:
            buckets[split_name].extend(by_family[family])

    splits = Splits(train=buckets["train"], dev=buckets["dev"], test=buckets["test"])
    assert_no_leakage(splits)
    return splits


def write_jsonl(rows: list[Utterance], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Utterance]:
    with path.open("r", encoding="utf-8") as fh:
        return [Utterance.from_dict(json.loads(line)) for line in fh if line.strip()]
