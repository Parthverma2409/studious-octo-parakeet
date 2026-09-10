"""FR10 -- the gate that decides whether a recommendation may be issued at all.

FR10 does not say "rank the candidates". It says a recommendation is issued only
when the margin and the sample size are sufficient. That word "only" is the whole
requirement, and the failure this module prevents is the comfortable one: a
comparison report that always names a winner. Such a report is indistinguishable
from a coin flip dressed up in a table, and it is worse than no report, because
somebody will act on it.

Three conditions, all of which must hold:

  1. **Sample size.** At least `MIN_PAIRWISE_COMPARISONS` (50) judged pairs. The
     66-scenario suite clears this; a family-level slice usually does not.
  2. **Margin beyond the interval.** The bootstrap 95% CI of the per-pair margin
     must exclude zero. "The margin exceeds the CI" and "the CI excludes zero"
     are the same statement, and the second is the one that is computable.
  3. **A usable judge.** If the judge's flip rate under A/B swapping exceeds
     `MAX_FLIP_RATE`, its win counts encode presentation order, so no margin
     computed from them means anything. `data/gold/README.md` requires the gate
     to account for this and this is where that happens.

Anything short of all three returns `decided=False` and the words "no
significant difference", with the reason attached.

    python -m packages.bench.recommend
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from packages.analysis.statistics import ConfidenceInterval, bootstrap_ci
from packages.bench.compare import PairwiseComparison

#: FR10's sample-size floor. Below this the bootstrap interval is being computed
#: from too few distinct pairs to resample meaningfully, whatever it reports.
MIN_PAIRWISE_COMPARISONS = 50

#: Above this flip rate the judge is reading position rather than quality. 0.20
#: is a deliberately generous ceiling: published LLM-judge position-bias rates
#: sit well above it, so a judge that passes this bar has been shown something.
MAX_FLIP_RATE = 0.20


@dataclass(frozen=True)
class Recommendation:
    """The verdict, and every reason it could have been withheld.

    `winner` is None whenever `decided` is False, and the two are never allowed
    to disagree -- a caller that reads only `winner` still cannot accidentally
    report an undecided comparison as a win.
    """

    decided: bool
    winner: str | None
    a_name: str
    b_name: str
    judge_name: str
    n: int
    margin: float
    interval: ConfidenceInterval
    flip_rate: float
    reasons: tuple[str, ...]

    def format(self) -> str:
        lines = [
            "candidates    : %s vs %s   judge=%s" % (self.a_name, self.b_name, self.judge_name),
            "pairs judged  : %d  (FR10 floor %d)" % (self.n, MIN_PAIRWISE_COMPARISONS),
            "margin        : %s" % self.interval.format(),
            "judge flips   : %.3f  (ceiling %.2f)" % (self.flip_rate, MAX_FLIP_RATE),
        ]
        if self.decided:
            lines.append("VERDICT       : recommend %s" % self.winner)
        else:
            lines.append("VERDICT       : NO SIGNIFICANT DIFFERENCE -- no recommendation issued")
        for reason in self.reasons:
            lines.append("                - %s" % reason)
        return "\n".join(lines)


def recommend(
    comparison: PairwiseComparison,
    min_pairs: int = MIN_PAIRWISE_COMPARISONS,
    max_flip_rate: float = MAX_FLIP_RATE,
    level: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 20260901,
) -> Recommendation:
    """Applies the FR10 gate to one pairwise comparison.

    The margin is the mean of the per-pair scores (+1 A, -1 B, 0 tie or flip),
    and its interval is bootstrapped over those pairs. Resampling the pairs
    rather than assuming a normal approximation matters here because the score
    takes three values: with many ties the sampling distribution is nothing like
    a normal one, and a t-interval over it would be too narrow in exactly the
    situation -- near-tied candidates -- where the gate has to hold.
    """
    reasons: list[str] = []

    per_pair = [float(x) for x in comparison.per_pair]
    if not per_pair:
        raise ValueError("cannot recommend from an empty comparison")

    interval = bootstrap_ci(per_pair, n_resamples=n_resamples, level=level, seed=seed)

    enough_pairs = comparison.n >= min_pairs
    if not enough_pairs:
        reasons.append("only %d pairwise comparisons; FR10 requires at least %d" % (comparison.n, min_pairs))

    margin_clears_ci = interval.excludes_zero()
    if not margin_clears_ci:
        reasons.append(
            "the %.0f%% CI for the margin is [%.4f, %.4f] and contains zero: the observed "
            "%+.4f is inside the noise" % (level * 100, interval.low, interval.high, interval.point)
        )

    judge_usable = comparison.flip_rate <= max_flip_rate
    if not judge_usable:
        reasons.append(
            "judge flip rate %.3f exceeds %.2f: it changes its answer when A and B are swapped, "
            "so its win counts measure position rather than quality"
            % (comparison.flip_rate, max_flip_rate)
        )

    decided = enough_pairs and margin_clears_ci and judge_usable
    winner = None
    if decided:
        winner = comparison.a_name if interval.point > 0 else comparison.b_name
        reasons.append(
            "margin %+.4f with %.0f%% CI [%.4f, %.4f] over %d pairs, judge flip rate %.3f"
            % (interval.point, level * 100, interval.low, interval.high, comparison.n, comparison.flip_rate)
        )

    return Recommendation(
        decided=decided,
        winner=winner,
        a_name=comparison.a_name,
        b_name=comparison.b_name,
        judge_name=comparison.judge_name,
        n=comparison.n,
        margin=interval.point,
        interval=interval,
        flip_rate=comparison.flip_rate,
        reasons=tuple(reasons),
    )


def main(argv: list[str] | None = None) -> int:
    """Demonstrates the gate on the three cases that matter, including the two
    it must refuse. A gate only ever exercised on the case it passes has not
    been tested."""
    from packages.bench.compare import (
        PositionBiasedJudge,
        RubricScoreJudge,
        pairwise_sweep,
        rule_based_configuration,
        run_configuration,
        synthetic_degraded_configuration,
    )
    from packages.bench.scenarios import load_scenarios

    parser = argparse.ArgumentParser(prog="packages.bench.recommend")
    parser.parse_args(argv)

    scenarios = load_scenarios()
    baseline = rule_based_configuration()
    degraded = synthetic_degraded_configuration(rate=0.5)
    a_results = run_configuration(baseline, scenarios)
    b_results = run_configuration(degraded, scenarios)
    judge = RubricScoreJudge()

    print("FR10 recommendation gate")
    print("a recommendation requires: n >= %d pairs AND a 95%% CI excluding zero AND a judge"
          % MIN_PAIRWISE_COMPARISONS)
    print("flip rate <= %.2f. Failing any one of them, the answer is 'no significant difference'." % MAX_FLIP_RATE)
    print("")

    cases = [
        (
            "1. genuine tie (a configuration against an identical copy)",
            pairwise_sweep(judge, a_results, a_results, "rule-based-v1", "rule-based-v1-copy"),
        ),
        (
            "2. real margin but too few pairs (first 12 scenarios only)",
            pairwise_sweep(judge, a_results[:12], b_results[:12], "rule-based-v1", "synthetic-degraded"),
        ),
        (
            "3. a judge that reads position (negative control)",
            pairwise_sweep(PositionBiasedJudge(), a_results, b_results, "rule-based-v1", "synthetic-degraded"),
        ),
        (
            "4. full suite, order-independent judge",
            pairwise_sweep(judge, a_results, b_results, "rule-based-v1", "synthetic-degraded"),
        ),
    ]

    for title, comparison in cases:
        print(title)
        for line in recommend(comparison).format().splitlines():
            print("  " + line)
        print("")

    print("REAL: the arithmetic, the gate and the flip rates. NOT REAL: the opponent. No Ollama")
    print("is installed, so 'synthetic-degraded' is a fault-injected copy of the rule-based run,")
    print("not a model. Case 4 shows the gate firing on a difference that was put there by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
