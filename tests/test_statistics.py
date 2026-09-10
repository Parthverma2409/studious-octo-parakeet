"""Statistics checked against values computed by hand, not against themselves.

A bug here is the worst kind of bug in an evaluation project: every number in the
final report inherits it and nothing else in the system misbehaves. So each test
states the expected value and where it comes from -- a closed-form derivation, a
textbook worked example, or an argument from the definition. Comparing a function
to another run of itself would pass just as happily with the arithmetic inverted.
"""

from __future__ import annotations

import math

import pytest

from packages.analysis.statistics import (
    auc_roc,
    bootstrap_ci,
    bootstrap_paired_diff_ci,
    cohens_kappa,
    krippendorff_alpha,
    mean,
    pearson_r,
    percentile,
    precision_at_recall,
)


# -- percentile ------------------------------------------------------------


def test_percentile_interpolates_between_neighbours() -> None:
    """Median of [1, 2, 3, 4] is 2.5: position (n-1)*0.5 = 1.5, halfway between
    the values at index 1 and 2. Expected 2.5."""
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_percentile_endpoints_are_the_extremes() -> None:
    """p0 and p100 of [10, 20, 30] are the min and max: 10 and 30."""
    assert percentile([30, 10, 20], 0) == 10
    assert percentile([30, 10, 20], 100) == 30


def test_percentile_p95_of_one_to_hundred_is_hand_computable() -> None:
    """For 1..100, position = 99 * 0.95 = 94.05, between values[94] = 95 and
    values[95] = 96, so 95 + 0.05 * 1 = 95.05."""
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_percentile_rejects_an_out_of_range_quantile() -> None:
    """A q of 0.95 instead of 95 is a plausible typo that would silently return
    close to the minimum; it must raise instead."""
    with pytest.raises(ValueError):
        percentile([1, 2, 3], 101)


# -- bootstrap -------------------------------------------------------------


def test_bootstrap_of_a_constant_sample_has_zero_width() -> None:
    """Every resample of [4, 4, 4, 4] has mean 4, so the CI is exactly [4, 4].
    A non-degenerate interval here would mean the resampling is inventing data."""
    ci = bootstrap_ci([4.0, 4.0, 4.0, 4.0], n_resamples=200, seed=1)
    assert ci.point == 4.0
    assert (ci.low, ci.high) == (4.0, 4.0)
    assert ci.half_width == 0.0


def test_bootstrap_of_two_points_spans_both() -> None:
    """With values [0, 1] a resample mean can only be 0, 0.5 or 1, with
    probabilities 1/4, 1/2, 1/4. The 2.5th percentile of that distribution is 0
    and the 97.5th is 1, so the 95% CI is exactly [0.0, 1.0] with point 0.5."""
    ci = bootstrap_ci([0.0, 1.0], n_resamples=4000, seed=7)
    assert ci.point == 0.5
    assert (ci.low, ci.high) == (0.0, 1.0)


def test_bootstrap_is_deterministic_for_a_given_seed() -> None:
    """A published interval that changes between two runs of the same script
    cannot be defended, so the same seed must reproduce the same bounds."""
    values = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2]
    first = bootstrap_ci(values, n_resamples=500, seed=42)
    second = bootstrap_ci(values, n_resamples=500, seed=42)
    third = bootstrap_ci(values, n_resamples=500, seed=43)
    assert (first.low, first.high) == (second.low, second.high)
    assert (first.low, first.high) != (third.low, third.high)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    """The percentile method cannot place the observed statistic outside its own
    resampling distribution: mean([1..9]) = 5 must lie inside the interval."""
    ci = bootstrap_ci([float(i) for i in range(1, 10)], n_resamples=1000, seed=3)
    assert ci.point == 5.0
    assert ci.low <= 5.0 <= ci.high


def test_paired_difference_of_identical_samples_is_exactly_zero() -> None:
    """a - b is the zero vector, so every resample mean is 0 and the CI is
    [0, 0]. This is the tie case the FR10 gate must refuse to call."""
    a = [1.0, 0.5, 0.75, 0.25]
    ci = bootstrap_paired_diff_ci(a, list(a), n_resamples=200, seed=5)
    assert (ci.low, ci.point, ci.high) == (0.0, 0.0, 0.0)
    assert not ci.excludes_zero()


def test_paired_difference_of_a_constant_offset_excludes_zero() -> None:
    """Every pair differs by exactly +0.2, so the difference sample is constant
    and the CI is [0.2, 0.2] -- which excludes zero."""
    a = [0.9, 0.7, 0.5, 0.3]
    b = [0.7, 0.5, 0.3, 0.1]
    ci = bootstrap_paired_diff_ci(a, b, n_resamples=200, seed=5)
    assert ci.point == pytest.approx(0.2)
    assert ci.excludes_zero()


def test_paired_difference_rejects_mismatched_lengths() -> None:
    """Pairing 66 scenarios against 65 is a silent off-by-one in a sweep; it
    must fail loudly rather than zip the shorter list."""
    with pytest.raises(ValueError):
        bootstrap_paired_diff_ci([1.0, 2.0], [1.0])


# -- Cohen's kappa ---------------------------------------------------------


def test_cohens_kappa_matches_the_worked_two_by_two_example() -> None:
    """Textbook 2x2: 20 both-yes, 5 A-yes/B-no, 10 A-no/B-yes, 15 both-no, n=50.
    Po = 35/50 = 0.70. Marginals: A yes 25/50, B yes 30/50, so
    Pe = 0.5*0.6 + 0.5*0.4 = 0.50. kappa = (0.70-0.50)/(1-0.50) = 0.40."""
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.40)


def test_cohens_kappa_is_one_for_perfect_agreement_with_two_categories() -> None:
    """Identical label vectors using both categories: Po = 1, so kappa = 1
    regardless of Pe."""
    labels = ["y", "n", "y", "n", "y"]
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_cohens_kappa_is_zero_at_chance_level() -> None:
    """A = [y,y,n,n], B = [y,n,y,n]: Po = 0.5. Both marginals are 2/4 yes, so
    Pe = 0.25 + 0.25 = 0.5 and kappa = (0.5-0.5)/(1-0.5) = 0. This is the case
    raw agreement reports as a respectable 50%."""
    assert cohens_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == pytest.approx(0.0)


def test_cohens_kappa_goes_negative_below_chance() -> None:
    """Perfect disagreement on balanced marginals: Po = 0, Pe = 0.5,
    kappa = (0 - 0.5)/0.5 = -1.0."""
    assert cohens_kappa(["y", "n"], ["n", "y"]) == pytest.approx(-1.0)


def test_cohens_kappa_flags_the_single_category_degenerate_case() -> None:
    """Both raters answered "success" on everything. Agreement is total but
    carries no information; the documented return is 1.0, and callers are
    required to report it as degenerate rather than as reliability."""
    assert cohens_kappa(["y"] * 6, ["y"] * 6) == 1.0


# -- Krippendorff's alpha --------------------------------------------------


def test_krippendorff_nominal_matches_the_closed_form_for_two_coders() -> None:
    """Units (0,0), (0,0), (1,1), (0,1): N = 8 values, one disagreeing unit.
    Do = 2/8 = 0.25. Pooled marginals n0 = 5, n1 = 3, so
    De = (5*3 + 3*5)/(8*7) = 30/56. alpha = 1 - 0.25/(30/56) = 1 - 14/30 = 8/15.
    Derived independently from the definition alpha = 1 - Do/De."""
    units = [(0, 0), (0, 0), (1, 1), (0, 1)]
    assert krippendorff_alpha(units, "nominal") == pytest.approx(8 / 15)


def test_krippendorff_interval_matches_a_hand_computed_coincidence_matrix() -> None:
    """Units (1,1), (2,2), (3,3), (1,2). N = 8, marginals n1 = 3, n2 = 3, n3 = 2.
    Do = (1*(1-2)^2 + 1*(2-1)^2)/8 = 0.25.
    De = [2*(3*3*1) + 2*(3*2*4) + 2*(3*2*1)] / (8*7) = 78/56.
    alpha = 1 - 0.25/(78/56) = 1 - 7/39 = 32/39 = 0.820513."""
    units = [(1, 1), (2, 2), (3, 3), (1, 2)]
    assert krippendorff_alpha(units, "interval") == pytest.approx(32 / 39)


def test_krippendorff_ordinal_matches_a_hand_computed_coincidence_matrix() -> None:
    """Same units as the interval case. Ordinal deltas from the marginals
    n1 = 3, n2 = 3, n3 = 2:
      d(1,2) = (3+3 - (3+3)/2)^2 = 9
      d(1,3) = (3+3+2 - (3+2)/2)^2 = 5.5^2 = 30.25
      d(2,3) = (3+2 - (3+2)/2)^2 = 2.5^2 = 6.25
    Do = 2*9/8 = 2.25. De = 2*(9*9 + 6*30.25 + 6*6.25)/56 = 600/56.
    alpha = 1 - 2.25/(600/56) = 1 - 0.21 = 0.79 exactly."""
    units = [(1, 1), (2, 2), (3, 3), (1, 2)]
    assert krippendorff_alpha(units, "ordinal") == pytest.approx(0.79)


def test_krippendorff_ordinal_and_nominal_differ_on_distant_disagreement() -> None:
    """Nominal cannot see that 1-against-5 is a worse disagreement than
    1-against-2, which is exactly why the 1-5 axes are scored ordinally. The two
    coefficients must not coincide on data where the distance matters."""
    units = [(1, 1), (2, 2), (5, 5), (1, 5)]
    assert krippendorff_alpha(units, "nominal") != pytest.approx(krippendorff_alpha(units, "ordinal"))


def test_krippendorff_ignores_units_only_one_coder_labelled() -> None:
    """A unit with a single value carries no pairable observation, so appending
    (1, None) and (None, None) to the interval example must leave alpha at 32/39.
    One coder's opinion is not evidence about agreement."""
    units = [(1, 1), (2, 2), (3, 3), (1, 2)]
    padded = units + [(1, None), (None, None)]
    assert krippendorff_alpha(padded, "interval") == pytest.approx(krippendorff_alpha(units, "interval"))
    assert krippendorff_alpha(padded, "interval") == pytest.approx(32 / 39)


def test_krippendorff_handles_three_coders_with_a_gap() -> None:
    """Units (1,1,1), (0,0,None), (1,0,1). The three-value units are weighted
    1/(m-1) = 1/2 per ordered pair, the two-value unit by 1.
    o11 = 3 + 1 = 4, o00 = 2, o01 = o10 = 1, N = 8, n1 = 5, n0 = 3.
    Do = 2/8 = 0.25, De = 30/56, alpha = 8/15 = 0.53333.
    This is the shape Cohen's kappa cannot express: three coders, missing cells."""
    units = [(1, 1, 1), (0, 0, None), (1, 0, 1)]
    assert krippendorff_alpha(units, "nominal") == pytest.approx(8 / 15)


def test_krippendorff_is_one_for_perfect_agreement() -> None:
    """Three coders agreeing on every unit while using both categories gives
    Do = 0, hence alpha = 1."""
    units = [(1, 1, 1), (0, 0, 0), (1, 1, 1), (0, 0, 0)]
    assert krippendorff_alpha(units, "nominal") == pytest.approx(1.0)


def test_krippendorff_rejects_an_unknown_level_of_measurement() -> None:
    """A typo like "ratio" must not silently fall through to nominal and quietly
    change every reported alpha."""
    with pytest.raises(ValueError):
        krippendorff_alpha([(1, 1), (1, 2)], "ratio")


# -- AUC and precision@recall ---------------------------------------------


def test_auc_matches_the_standard_worked_example() -> None:
    """scores [0.1, 0.4, 0.35, 0.8] with labels [0, 0, 1, 1]. Of the four
    positive/negative pairs the positives win three (0.35 beats 0.1; 0.8 beats
    both) and lose one (0.35 against 0.4), so AUC = 3/4 = 0.75."""
    assert auc_roc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]) == pytest.approx(0.75)


def test_auc_of_a_perfect_ranking_is_one() -> None:
    """Every positive scores above every negative: 4/4 pairs won, AUC = 1.0."""
    assert auc_roc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_auc_of_an_inverted_ranking_is_zero() -> None:
    """The same data with the labels swapped loses every pair: AUC = 0.0. A
    detector at 0.0 is as informative as one at 1.0, and an implementation that
    clamped to 0.5 would hide that."""
    assert auc_roc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auc_counts_a_tie_as_half_a_pair() -> None:
    """scores [1, 1, 0], labels [1, 0, 0]: the positive ties one negative
    (worth 0.5) and beats the other (worth 1), so AUC = 1.5/2 = 0.75. Tied
    scores are the normal case for a rubric evaluator with few distinct outputs."""
    assert auc_roc([1.0, 1.0, 0.0], [1, 0, 0]) == pytest.approx(0.75)


def test_auc_of_a_constant_score_is_exactly_one_half() -> None:
    """Every score identical: every pair is a tie, AUC = 0.5. This is what "no
    signal" looks like, and RQ1 must be able to report it."""
    assert auc_roc([0.5] * 6, [1, 1, 1, 0, 0, 0]) == pytest.approx(0.5)


def test_auc_raises_when_a_class_is_missing() -> None:
    """With no negatives the quantity is undefined. Returning 0.5 instead would
    be indistinguishable, in a results table, from a real finding of no signal."""
    with pytest.raises(ValueError):
        auc_roc([0.1, 0.2, 0.3], [1, 1, 1])


def test_precision_at_full_recall_is_hand_computable() -> None:
    """scores [0.9, 0.8, 0.7, 0.6], labels [1, 0, 1, 0]. Catching both positives
    needs the top three, of which two are positive: precision 2/3 = 0.667."""
    assert precision_at_recall([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], 1.0) == pytest.approx(2 / 3)


def test_precision_at_half_recall_takes_the_best_qualifying_point() -> None:
    """The same data at recall 0.5: the top-1 cut already reaches recall 0.5 with
    precision 1.0, which is the best qualifying operating point."""
    assert precision_at_recall([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], 0.5) == pytest.approx(1.0)


def test_precision_at_recall_of_a_perfect_ranker_is_one() -> None:
    """Positives ranked strictly above negatives: full recall at precision 1.0."""
    assert precision_at_recall([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], 1.0) == pytest.approx(1.0)


# -- Pearson r -------------------------------------------------------------


def test_pearson_is_one_for_an_exact_positive_linear_relation() -> None:
    """y = 2x is perfectly correlated: r = 1.0."""
    assert pearson_r([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_pearson_is_minus_one_for_an_exact_negative_relation() -> None:
    """y = -x: r = -1.0."""
    assert pearson_r([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_matches_a_hand_computed_partial_correlation() -> None:
    """x = [1,2,3,4], y = [2,4,5,4]. Means 2.5 and 3.75.
    Sxy = (-1.5)(-1.75) + (-0.5)(0.25) + (0.5)(1.25) + (1.5)(0.25) = 3.5
    Sxx = 5.0, Syy = 4.75, r = 3.5/sqrt(23.75) = 0.718185."""
    assert pearson_r([1, 2, 3, 4], [2, 4, 5, 4]) == pytest.approx(0.718185, abs=1e-6)


def test_pearson_returns_nan_for_a_constant_series() -> None:
    """A judge that gave every response the same score has an undefined
    correlation with length, not a zero one. Reporting 0.0 would read as
    "verbosity bias checked, none found"."""
    assert math.isnan(pearson_r([1, 1, 1, 1], [1, 2, 3, 4]))


# -- guards ----------------------------------------------------------------


def test_mean_of_an_empty_sample_raises() -> None:
    """Zero scenarios is a broken sweep, not a mean of zero."""
    with pytest.raises(ValueError):
        mean([])


def test_auc_rejects_mismatched_lengths() -> None:
    """Scores and labels drifting out of step silently mislabels the study."""
    with pytest.raises(ValueError):
        auc_roc([0.1, 0.2], [1])
