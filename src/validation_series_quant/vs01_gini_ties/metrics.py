from __future__ import annotations

from typing import Iterable, Tuple, Union

import numpy as np
from sklearn.metrics import roc_auc_score

ArrayLike = Union[np.ndarray, Iterable[float]]


def _prepare_inputs(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Validate and sanitize inputs; returns (y:int, s:float, n_pos, n_neg)."""
    y = np.asarray(y_true)
    s = np.asarray(y_score)

    if y.ndim != 1 or s.ndim != 1:
        raise ValueError(f"y_true and y_score must be 1D, got shapes {y.shape} and {s.shape}")
    if y.shape[0] != s.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    # Convert score to float early (for finite checks / sorting / unique)
    s = s.astype(float, copy=False)

    if dropna:
        m = np.isfinite(s) & np.isfinite(y.astype(float, copy=False))
        y = y[m]
        s = s[m]

    # y must be {0,1} (allow bool/int; reject non-integer floats)
    if np.issubdtype(y.dtype, np.floating):
        if not np.all(np.isfinite(y)):
            raise ValueError("y_true contains non-finite values")
        if not np.all(np.equal(y, np.floor(y))):
            raise ValueError("y_true must be integer/bool (0/1), got non-integers")
    y = y.astype(int, copy=False)

    u = np.unique(y)
    if not np.all(np.isin(u, [0, 1])):
        raise ValueError(f"y_true must be binary in {{0,1}}, got unique={u.tolist()}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive (1) and one negative (0) sample")

    return y, s, n_pos, n_neg


def _gini_from_auc(auc: float) -> float:
    """Convert AUC to Gini: G = 2*AUC - 1."""
    return 2.0 * float(auc) - 1.0


def auc_standard(y_true: ArrayLike, y_score: ArrayLike, *, dropna: bool = True) -> float:
    """Standard ROC AUC (ties treated neutrally as 0.5 in sklearn)."""
    y, s, _, _ = _prepare_inputs(y_true, y_score, dropna=dropna)
    return float(roc_auc_score(y, s))


def conservative_ties_correction(
        y_true: ArrayLike,
        y_score: ArrayLike,
        *,
        dropna: bool = True
) -> float:
    """
    Conservative ties correction based on tie groups:

        T = sum_j pos_j * neg_j  over groups with equal score
        corr = T / (N_pos * N_neg)

    Depends only on tie structure (permutation-invariant).
    """
    y, s, n_pos, n_neg = _prepare_inputs(y_true, y_score, dropna=dropna)

    # group by identical scores
    _, inv = np.unique(s, return_inverse=True)
    pos_j = np.bincount(inv, weights=y.astype(float))
    cnt_j = np.bincount(inv)
    neg_j = cnt_j - pos_j

    ties_pairs = float(np.sum(pos_j * neg_j))
    return ties_pairs / (float(n_pos) * float(n_neg))


def gini_auc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = True,
    conservative_ties: bool = False,
) -> float:
    """
    Gini via ROC AUC.

    If conservative_ties=True:
        gini = (2*AUC - 1) - correction
    where correction depends only on tie structure.
    """
    auc = auc_standard(y_true, y_score, dropna=dropna)
    g = _gini_from_auc(auc)
    if conservative_ties:
        g -= conservative_ties_correction(y_true, y_score, dropna=dropna)
    return float(g)


def _cap_gini_from_sorted_labels(y_sorted: np.ndarray, n_pos: int) -> float:
    """CAP-based Gini assuming y_sorted is ordered by descending score (ties already resolved)."""
    N = y_sorted.size

    x = np.arange(N + 1, dtype=float) / float(N)
    cum_pos = np.zeros(N + 1, dtype=float)
    cum_pos[1:] = np.cumsum(y_sorted == 1, dtype=float)
    y = cum_pos / float(n_pos)

    area_model = float(np.trapz(y, x))
    area_random = 0.5

    p = n_pos / float(N)
    area_perfect = 1.0 - 0.5 * p  # perfect CAP area

    return (area_model - area_random) / (area_perfect - area_random)


def gini_cap(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    dropna: bool = True,
    tie_break: str = "stable",
) -> float:
    """
    Gini via CAP curve (trapezoid).

    tie_break:
      - "stable": stable sort by score desc (deterministic but tie order follows input order)
      - "conservative": within ties, put y=0 first then y=1 (worst-case if 1 is “positive/bad”)
      - "optimistic": within ties, put y=1 first then y=0
    """
    if tie_break not in {"stable", "conservative", "optimistic"}:
        raise ValueError("tie_break must be one of: stable, conservative, optimistic")

    y, s, n_pos, _ = _prepare_inputs(y_true, y_score, dropna=dropna)

    if tie_break == "stable":
        order = np.argsort(-s, kind="mergesort")
    else:
        # lexsort: primary key is last; sort by -score then by y_key
        y_key = y if tie_break == "conservative" else (1 - y)
        order = np.lexsort((y_key, -s))

    return float(_cap_gini_from_sorted_labels(y[order], n_pos=n_pos))
