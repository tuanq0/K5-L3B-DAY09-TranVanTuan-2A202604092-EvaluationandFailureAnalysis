"""Statistics students implement (M2 judge calibration, M4 paired bootstrap).

Rules: a judge is a classifier -- measure it like one. A higher mean is not
"better" until the paired CI of the difference excludes 0.
Reference values to self-check are in tests/test_lab9_stats.py.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

POS = "FAIL"   # the judge's job is to FIND failures, so FAIL is the positive class


def confusion(human: Sequence[str], judge: Sequence[str]) -> dict[str, int]:
    """tp/fp/fn/tn with FAIL as positive."""
    # Pair labels by index: matching lengths are required to avoid silently
    # comparing or dropping observations.
    if len(human) != len(judge):
        raise ValueError("human and judge labels must have the same length")

    result = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for expected, predicted in zip(human, judge):
        if predicted == POS:
            key = "tp" if expected == POS else "fp"
        else:
            key = "fn" if expected == POS else "tn"
        result[key] += 1
    return result


def precision_recall(human: Sequence[str], judge: Sequence[str]) -> tuple[float, float]:
    # TODO M2: precision = tp/(tp+fp), recall = tp/(tp+fn); return 0.0 when the denominator is 0
    counts = confusion(human, judge)
    precision_denominator = counts["tp"] + counts["fp"]
    recall_denominator = counts["tp"] + counts["fn"]
    precision = counts["tp"] / precision_denominator if precision_denominator else 0.0
    recall = counts["tp"] / recall_denominator if recall_denominator else 0.0
    return precision, recall


def cohen_kappa(human: Sequence[str], judge: Sequence[str]) -> float:
    """kappa = (p_o - p_e) / (1 - p_e). Raise ValueError when undefined (p_e == 1):
    a constant judge has no signal and must FAIL calibration, not pass on a technicality."""
    counts = confusion(human, judge)
    n = len(human)
    if n == 0:
        raise ValueError("kappa is undefined for an empty label set")

    p_observed = (counts["tp"] + counts["tn"]) / n
    human_fail = counts["tp"] + counts["fn"]
    judge_fail = counts["tp"] + counts["fp"]
    human_pass = counts["tn"] + counts["fp"]
    judge_pass = counts["tn"] + counts["fn"]

    # Expected agreement by chance: probabilities of both classes agreeing.
    p_expected = (
        (human_fail / n) * (judge_fail / n)
        + (human_pass / n) * (judge_pass / n)
    )
    if p_expected == 1.0:
        raise ValueError("kappa is undefined when expected agreement is 1")
    return (p_observed - p_expected) / (1.0 - p_expected)


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Correct an observed failure rate for judge error; clip to [0, 1]."""
    denominator = sensitivity + specificity - 1.0
    if denominator == 0.0:
        raise ValueError("Rogan-Gladen correction is undefined when sensitivity + specificity - 1 is zero")
    corrected = (observed_rate + specificity - 1.0) / denominator
    return min(1.0, max(0.0, corrected))


def paired_bootstrap(a: Sequence[float], b: Sequence[float], *, n_resamples: int = 1000,
                     rng: int = 2027, confidence: float = 0.95) -> tuple[float, float, float]:
    """a[i], b[i] = per-item mean over seeds, SAME items, SAME order.
    Return (mean(a)-mean(b), lo, hi) with scipy.stats.bootstrap(paired=True, method="percentile",
    rng=numpy.random.default_rng(rng)). If a == b everywhere return (0.0, 0.0, 0.0)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    if not a:
        return 0.0, 0.0, 0.0
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    if all(x == y for x, y in zip(a, b)):
        return 0.0, 0.0, 0.0

    import numpy as np
    from scipy.stats import bootstrap

    a_values = np.asarray(a, dtype=float)
    b_values = np.asarray(b, dtype=float)
    result = bootstrap(
        (a_values, b_values),
        statistic=lambda x, y: float(np.mean(x) - np.mean(y)),
        paired=True,
        method="percentile",
        n_resamples=n_resamples,
        confidence_level=confidence,
        rng=np.random.default_rng(rng),
    )
    difference = float(np.mean(a_values) - np.mean(b_values))
    lo = float(result.confidence_interval.low)
    hi = float(result.confidence_interval.high)
    return difference, lo, hi


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability that AT LEAST ONE of k trials drawn from n (c of them passing) passes.
    Unbiased estimator (Chen et al. 2021): 1 - C(n-c, k) / C(n, k). Raise ValueError if k > n."""
    if n < 0 or c < 0 or k < 0:
        raise ValueError("n, c, and k must be non-negative")
    if c > n:
        raise ValueError("c cannot exceed n")
    if k > n:
        raise ValueError("k cannot exceed n")
    denominator = math.comb(n, k)
    if denominator == 0:
        return 0.0
    failing_draws = math.comb(n - c, k)
    return 1.0 - failing_draws / denominator


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k (tau-bench, Yao et al. 2024): probability that ALL k trials drawn from n pass.
    Unbiased estimator: C(c, k) / C(n, k). Raise ValueError if k > n.
    pass^k <= pass@1 <= pass@k; a reliable agent keeps pass^k close to pass@1."""
    if n < 0 or c < 0 or k < 0:
        raise ValueError("n, c, and k must be non-negative")
    if c > n:
        raise ValueError("c cannot exceed n")
    if k > n:
        raise ValueError("k cannot exceed n")
    denominator = math.comb(n, k)
    if denominator == 0:
        return 0.0
    return math.comb(c, k) / denominator
