import numpy as np

from validation_series_quant.vs01_gini_ties import metrics


def _toy_with_ties(seed: int = 0, n: int = 2000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    raw = 0.8 * y + rng.normal(0.0, 1.0, size=n)
    s = np.round(raw, 1)  # create ties
    return y, s


def test_correction_zero_without_ties():
    y = np.array([0, 1, 0, 1], dtype=int)
    s = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)  # all unique
    corr = metrics.conservative_ties_correction(y, s)
    assert corr == 0.0


def test_conservative_correction_is_permutation_invariant():
    y, s = _toy_with_ties(seed=1, n=5000)
    corr0 = metrics.conservative_ties_correction(y, s)
    rng = np.random.default_rng(123)
    for _ in range(20):
        idx = rng.permutation(len(y))
        corr = metrics.conservative_ties_correction(y[idx], s[idx])
        assert corr == corr0


def test_cap_tie_break_ordering():
    y, s = _toy_with_ties(seed=2, n=5000)
    g_cons = metrics.gini_cap(y, s, tie_break="conservative")
    g_stable = metrics.gini_cap(y, s, tie_break="stable")
    g_opt = metrics.gini_cap(y, s, tie_break="optimistic")
    assert g_cons <= g_stable <= g_opt


def test_gini_auc_conservative_is_lower_or_equal():
    y, s = _toy_with_ties(seed=3, n=5000)
    g_std = metrics.gini_auc(y, s, conservative_ties=False)
    g_cons = metrics.gini_auc(y, s, conservative_ties=True)
    assert g_cons <= g_std
