"""Inference statistics, implemented here rather than imported.

Every claim in RQ1, RQ2 and FR10 is a point estimate over a small sample: 66
scenarios, 150 triple-coded trajectories, a few hundred pairwise judgements. A
point estimate with no interval is not a result, and the failure this module
prevents is the one that looks like success -- reporting "config B scores 0.88
against config A's 0.85" when the difference is inside the noise, or reporting a
judge's agreement as a raw percentage when chance alone explains most of it.

Nothing here needs numpy. The samples are small enough that a bootstrap in pure
Python costs milliseconds, and a dependency-free implementation is one anyone
grading this can read end to end.

Implemented:
  * percentile / bootstrap confidence intervals (percentile method, seeded)
  * Cohen's kappa            -- judge against one adjudicated label set (RQ2)
  * Krippendorff's alpha     -- three coders, 150 trajectories, missing values (D5)
  * AUC (ROC), precision@recall -- detection of known-bad trajectories (RQ1)
  * Pearson r                -- verbosity bias: score against response length (RQ2)

Each is checked in `tests/test_statistics.py` against a value computed by hand or
published in the literature, never against another run of itself.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

#: Any function from a sample to a number. The sample elements need not be
#: floats: `bootstrap_ci` resamples whatever it is given, so an element can be
#: a whole observation -- (score_a, score_b, label) triples, for instance --
#: which is what lets RQ1 bootstrap a difference of two AUCs over the same
#: resampled trajectories rather than over two independent ones.
Statistic = Callable[[Sequence[Any]], float]


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sample is undefined")
    return sum(values) / len(values)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile, `q` in [0, 100].

    Matches numpy's default `linear` method so the numbers can be checked
    against a familiar tool even though numpy is not installed here.
    """
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100], got %r" % q)
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(ordered[int(pos)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate and the interval that decides whether it means anything."""

    point: float
    low: float
    high: float
    level: float
    n: int
    n_resamples: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def excludes_zero(self) -> bool:
        """True when the whole interval sits on one side of zero.

        This is the arithmetic behind the FR10 recommendation gate: a difference
        whose interval straddles zero is not a difference.
        """
        return self.low > 0.0 or self.high < 0.0

    def format(self) -> str:
        return "%.4f  [%.4f, %.4f] (%.0f%%, n=%d)" % (self.point, self.low, self.high, self.level * 100, self.n)


def bootstrap_ci(
    values: Sequence[Any],
    statistic: Statistic = mean,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 20260901,
) -> ConfidenceInterval:
    """Percentile-method bootstrap CI for any statistic of one sample.

    Seeded, so the same data always yields the same interval: a result that
    changes between two runs of the same script cannot be defended in a report.
    """
    if not values:
        raise ValueError("bootstrap of an empty sample is undefined")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1), got %r" % level)

    rng = random.Random(seed)
    n = len(values)
    point = statistic(values)
    if n == 1:
        return ConfidenceInterval(point, point, point, level, n, 0)

    estimates: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(statistic(sample))

    tail = (1.0 - level) / 2.0 * 100.0
    return ConfidenceInterval(
        point=point,
        low=percentile(estimates, tail),
        high=percentile(estimates, 100.0 - tail),
        level=level,
        n=n,
        n_resamples=n_resamples,
    )


def bootstrap_paired_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 20260901,
) -> ConfidenceInterval:
    """CI for the mean of (a - b) over paired observations.

    Pairing matters for FR10: both configurations run the *same* 66 scenarios,
    so scenario-to-scenario variance is shared and should not be charged against
    the difference. Treating the two runs as independent samples would widen the
    interval and hide real differences.
    """
    if len(a) != len(b):
        raise ValueError("paired samples must be the same length (%d vs %d)" % (len(a), len(b)))
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, mean, n_resamples=n_resamples, level=level, seed=seed)


def cohens_kappa(rater_a: Sequence[Any], rater_b: Sequence[Any]) -> float:
    """Chance-corrected agreement between exactly two raters (the RQ2 headline).

    Raw agreement is the number that flatters a judge. On the stratified gold
    set, answering "success" unconditionally agrees with the humans a third of
    the time for free; kappa subtracts that.

    Returns 1.0 for perfect agreement, 0.0 for chance level, negative for worse
    than chance. When both raters used a single category the correction is
    undefined and 1.0 is returned -- callers must report that as a degenerate
    case, not as perfect reliability.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same units (%d vs %d)" % (len(rater_a), len(rater_b)))
    n = len(rater_a)
    if n == 0:
        raise ValueError("kappa over zero units is undefined")

    observed = sum(1 for x, y in zip(rater_a, rater_b) if x == y) / n
    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    expected = sum((count_a[c] / n) * (count_b[c] / n) for c in set(count_a) | set(count_b))

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def _coincidence_matrix(units: Sequence[Sequence[Any]]) -> tuple[dict[tuple[Any, Any], float], Counter, float]:
    """Krippendorff's coincidence matrix over units holding two or more values.

    Units carrying a single value are dropped rather than imputed: one coder's
    opinion is not evidence about agreement.
    """
    coincidences: dict[tuple[Any, Any], float] = {}
    marginals: Counter = Counter()
    total = 0.0
    for unit in units:
        present = [v for v in unit if v is not None]
        m = len(present)
        if m < 2:
            continue
        weight = 1.0 / (m - 1)
        for i, c in enumerate(present):
            marginals[c] += 1.0
            total += 1.0
            for j, k in enumerate(present):
                if i != j:
                    coincidences[(c, k)] = coincidences.get((c, k), 0.0) + weight
    return coincidences, marginals, total


def _delta_squared(c: Any, k: Any, level: str, ordered: list[Any], marginals: Counter) -> float:
    if level == "nominal":
        return 0.0 if c == k else 1.0
    if level == "interval":
        return float(c - k) ** 2
    if level == "ordinal":
        lo, hi = sorted((ordered.index(c), ordered.index(k)))
        between = sum(marginals[ordered[g]] for g in range(lo, hi + 1))
        return (between - (marginals[c] + marginals[k]) / 2.0) ** 2
    raise ValueError("unknown level of measurement: %r" % level)


def krippendorff_alpha(units: Sequence[Sequence[Any]], level: str = "nominal") -> float:
    """Agreement among N coders with missing values allowed (D5 protocol step 2).

    `units` is unit-major: one inner sequence per trajectory holding each coder's
    label in a fixed order, with `None` where that coder did not label it. Three
    team members triple-coding 150 trajectories is exactly this shape, and
    Cohen's kappa cannot express it -- it handles two complete raters only.

    `level` selects the difference function: "nominal" for `task_success` and the
    other binary axes, "ordinal" for the 1-5 axes, because a 1-against-5
    disagreement is worse than 1-against-2 and nominal alpha cannot see that.

    The D5 protocol gates on alpha >= 0.67 per axis before the full 150 are
    coded, which is why this is computed at pilot time rather than at write-up.
    """
    coincidences, marginals, total = _coincidence_matrix(units)
    if total < 2:
        raise ValueError("need at least one unit carrying two or more labels")

    ordered = sorted(marginals)
    observed = 0.0
    for (c, k), count in coincidences.items():
        observed += count * _delta_squared(c, k, level, ordered, marginals)
    observed /= total

    expected = 0.0
    for c in ordered:
        for k in ordered:
            if c != k:
                expected += marginals[c] * marginals[k] * _delta_squared(c, k, level, ordered, marginals)
    expected /= total * (total - 1.0)

    if expected == 0.0:
        return 1.0
    return 1.0 - observed / expected


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, ties sharing their mean rank.

    Ties are the normal case here: a rubric evaluator emits a handful of distinct
    scores over 66 trajectories, and dropping tie handling would silently inflate
    or deflate AUC depending on the ordering of equal scores.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def auc_roc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, via the Mann-Whitney rank identity (RQ1 metric).

    Equal to the probability that a randomly chosen known-bad trajectory scores
    above a randomly chosen good one, ties counting a half. Chosen over accuracy
    because the RQ1 evaluators emit a graded suspicion score and no threshold has
    been agreed; AUC compares the two variants over every threshold at once.

    Raises when one class is absent rather than returning 0.5, because a silent
    0.5 in a results table is indistinguishable from a real finding of no signal.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length (%d vs %d)" % (len(scores), len(labels)))
    positives = sum(1 for y in labels if y)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC needs both classes present (got %d positive, %d negative)" % (positives, negatives))

    ranks = _average_ranks(scores)
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def precision_at_recall(scores: Sequence[float], labels: Sequence[int], target_recall: float) -> float:
    """Best precision among operating points reaching at least `target_recall`.

    AUC is threshold-free but abstract. The operational RQ1 question is concrete:
    if a reviewer must catch 80% of the bad trajectories, what fraction of the
    queue they read is actually bad? That is this number.
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels must be the same length (%d vs %d)" % (len(scores), len(labels)))
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1], got %r" % target_recall)
    positives = sum(1 for y in labels if y)
    if positives == 0:
        raise ValueError("precision@recall needs at least one positive")

    ordered = sorted(zip(scores, labels), key=lambda pair: -pair[0])
    best = 0.0
    hits = 0
    for i, (_score, y) in enumerate(ordered, start=1):
        hits += 1 if y else 0
        if hits / positives >= target_recall:
            best = max(best, hits / i)
    return best


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Correlation coefficient -- the RQ2 verbosity-bias check.

    A judge whose score tracks response length is rewarding output volume. The
    check is deliberately a correlation and not a significance test: at a few
    hundred judgements the effect size is what matters, and a p-value here would
    invite reading "not significant" as "no bias".

    Returns NaN when either series is constant. That is undefined, not zero, and
    callers must report it as "not computable" rather than "no bias found".
    """
    if len(x) != len(y):
        raise ValueError("x and y must be the same length (%d vs %d)" % (len(x), len(y)))
    if len(x) < 2:
        raise ValueError("correlation needs at least two observations")

    mx, my = mean(x), mean(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0.0 or syy == 0.0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)
