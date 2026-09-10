"""The FR10 gate: a recommendation is issued ONLY when it is earned.

The property protected here is the negative one, and it is the requirement.
FR10 says a recommendation is issued only when the margin and the sample size are
sufficient, so a system that always names a winner does not satisfy FR10 -- it
satisfies nothing, because its output carries no information. Most of these tests
therefore assert that no winner was named, which is what a correct gate does
most of the time.

Three ways to fail the gate, tested one at a time so a passing verdict cannot be
produced by two conditions cancelling out: too few pairs, an interval that
straddles zero, and a judge whose answers move with presentation order.
"""

from __future__ import annotations

import pytest

from packages.bench.compare import PairwiseComparison
from packages.bench.recommend import MAX_FLIP_RATE, MIN_PAIRWISE_COMPARISONS, recommend


def _comparison(per_pair: list[int], flips: int = 0, a: str = "config-a", b: str = "config-b") -> PairwiseComparison:
    return PairwiseComparison(
        a_name=a,
        b_name=b,
        judge_name="test-judge",
        n=len(per_pair),
        wins_a=per_pair.count(1),
        wins_b=per_pair.count(-1),
        ties=per_pair.count(0) - flips,
        flips=flips,
        per_pair=list(per_pair),
    )


# -- the case the requirement is about -------------------------------------


def test_a_tie_names_no_winner() -> None:
    """60 pairs, 25 wins each way and 10 ties: the margin is 0 and its bootstrap
    interval contains 0. The verdict must be "no significant difference" and the
    winner must be None -- not the configuration that happened to be passed
    first, and not a coin flip."""
    comparison = _comparison([1] * 25 + [-1] * 25 + [0] * 10)
    result = recommend(comparison)
    assert result.decided is False
    assert result.winner is None
    assert "NO SIGNIFICANT DIFFERENCE" in result.format()


def test_a_near_tie_with_plenty_of_pairs_still_names_no_winner() -> None:
    """A 2-pair edge over 60 comparisons (31 to 29) is the case that most invites
    a fabricated recommendation: the win counts differ, so a naive report would
    name a winner. The interval straddles zero, so this one must not."""
    comparison = _comparison([1] * 31 + [-1] * 29)
    result = recommend(comparison)
    assert result.decided is False
    assert result.winner is None
    assert any("contains zero" in reason for reason in result.reasons)


def test_an_undecided_result_always_has_a_null_winner() -> None:
    """The invariant that makes the gate safe to consume: a caller reading only
    `winner` can never mistake an undecided comparison for a win, whatever the
    reason for the refusal."""
    for comparison in (
        _comparison([1] * 25 + [-1] * 25 + [0] * 10),
        _comparison([1] * 20),
        _comparison([1] * 40 + [0] * 30, flips=30),
    ):
        result = recommend(comparison)
        assert result.decided is (result.winner is not None)
        if not result.decided:
            assert result.winner is None
            assert result.reasons, "a refusal with no stated reason is unreviewable"


# -- gate 1: sample size ---------------------------------------------------


def test_too_few_pairs_blocks_a_unanimous_result() -> None:
    """20 pairs, every one a win for A, so the interval is [1, 1] and excludes
    zero. FR10's sample-size floor still refuses: 20 < 50."""
    result = recommend(_comparison([1] * 20))
    assert result.decided is False
    assert result.winner is None
    assert any("at least %d" % MIN_PAIRWISE_COMPARISONS in reason for reason in result.reasons)


def test_the_sample_size_floor_is_inclusive_at_exactly_fifty() -> None:
    """Exactly 50 pairs satisfies "n >= 50"; 49 does not. Pinned from both sides
    because an off-by-one here changes which comparisons are reportable."""
    assert recommend(_comparison([1] * 50)).decided is True
    assert recommend(_comparison([1] * 49)).decided is False


# -- gate 2: the interval --------------------------------------------------


def test_a_wide_margin_over_enough_pairs_is_recommended() -> None:
    """The positive control. 60 pairs, 45 wins for A and 15 ties, no flips: the
    interval clears zero and a recommendation is issued naming A. A gate that
    never recommends anything is as useless as one that always does."""
    result = recommend(_comparison([1] * 45 + [0] * 15))
    assert result.decided is True
    assert result.winner == "config-a"
    assert "recommend config-a" in result.format()


def test_the_losing_configuration_can_win() -> None:
    """The verdict follows the sign of the margin, not the argument order: with B
    ahead, B is named."""
    result = recommend(_comparison([-1] * 45 + [0] * 15))
    assert result.decided is True
    assert result.winner == "config-b"


def test_the_interval_is_reported_whether_or_not_a_winner_is_named() -> None:
    """A refusal has to show its arithmetic, or nobody can tell a genuine tie
    from a broken sweep."""
    result = recommend(_comparison([1] * 25 + [-1] * 25 + [0] * 10))
    assert result.interval.n == 60
    assert result.interval.contains(0.0)
    assert "margin" in result.format()


# -- gate 3: the judge -----------------------------------------------------


def test_a_position_reading_judge_blocks_the_recommendation() -> None:
    """70 pairs with a clear 40-win margin, but the judge changed its answer on
    30 of them when A and B were swapped -- a flip rate of 0.43 against a 0.20
    ceiling. Its win counts encode presentation order, so the margin computed
    from them means nothing and no recommendation may be issued."""
    comparison = _comparison([1] * 40 + [0] * 30, flips=30)
    assert comparison.flip_rate > MAX_FLIP_RATE
    result = recommend(comparison)
    assert result.decided is False
    assert result.winner is None
    assert any("flip rate" in reason for reason in result.reasons)


def test_a_small_flip_rate_within_the_ceiling_does_not_block() -> None:
    """Some flipping is tolerated -- the ceiling is a threshold, not a demand for
    perfection. 5 flips in 65 pairs is 0.077, inside the 0.20 ceiling."""
    comparison = _comparison([1] * 45 + [0] * 20, flips=5)
    assert comparison.flip_rate < MAX_FLIP_RATE
    assert recommend(comparison).decided is True


def test_every_failed_gate_is_reported_not_just_the_first() -> None:
    """A comparison that fails on sample size and on the judge must say both.
    Reporting only the first would send someone off to collect more pairs from a
    judge whose answers move with the order."""
    result = recommend(_comparison([1] * 10 + [0] * 10, flips=10))
    assert result.decided is False
    assert len(result.reasons) >= 2


# -- guards ----------------------------------------------------------------


def test_an_empty_comparison_raises_rather_than_returning_a_tie() -> None:
    """Zero judged pairs is a broken sweep. Reporting it as "no significant
    difference" would make a crashed run indistinguishable from a real tie."""
    with pytest.raises(ValueError):
        recommend(_comparison([]))


def test_the_verdict_is_stable_across_runs() -> None:
    """The bootstrap is seeded, so the same comparison yields the same interval
    and the same verdict. A recommendation that changes between two runs of the
    same data cannot be defended."""
    comparison = _comparison([1] * 30 + [-1] * 20 + [0] * 10)
    first, second = recommend(comparison), recommend(comparison)
    assert (first.decided, first.winner) == (second.decided, second.winner)
    assert (first.interval.low, first.interval.high) == (second.interval.low, second.interval.high)


def test_the_gate_refuses_on_the_real_suite_when_both_sides_are_identical() -> None:
    """End to end, with no synthetic numbers: the rule-based configuration judged
    against a copy of itself over all 66 scenarios. n clears the floor and the
    judge does not flip, so the only thing standing between this and a fabricated
    winner is the interval -- and it must hold."""
    from packages.bench.compare import (
        RubricScoreJudge,
        pairwise_sweep,
        rule_based_configuration,
        run_configuration,
    )
    from packages.bench.scenarios import load_scenarios

    results = run_configuration(rule_based_configuration(), load_scenarios())
    comparison = pairwise_sweep(RubricScoreJudge(), results, results, "rule-based", "rule-based-copy")
    assert comparison.n >= MIN_PAIRWISE_COMPARISONS
    assert comparison.flip_rate == 0.0
    verdict = recommend(comparison)
    assert verdict.decided is False
    assert verdict.winner is None
