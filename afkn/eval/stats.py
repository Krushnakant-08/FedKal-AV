"""Paired statistics across seeds (torch-free): Wilcoxon signed-rank with a rank-biserial effect size."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_std(values: list[float] | np.ndarray) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1; 0 for one value) and count of the finite values."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def paired_wilcoxon(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> dict[str, float]:
    """Paired two-sided Wilcoxon signed-rank test of ``a`` vs ``b`` (same seeds, same order).

    Returns n pairs, statistic, p, median difference (a - b) and the matched-pairs rank-biserial correlation
    ``r = (W+ - W-) / (W+ + W-)`` over the non-zero differences (r > 0: ``a`` tends to be larger). With fewer than
    6 non-zero pairs the exact two-sided p cannot go below 0.05, so treat p as descriptive only (``n_nonzero`` is
    returned for that check). NaN where undefined (no non-zero differences).
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have the same length")
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[ok]
    nz = d[d != 0]
    out = {"n": float(ok.sum()), "n_nonzero": float(nz.size), "statistic": float("nan"), "p": float("nan"),
           "median_diff": float(np.median(d)) if d.size else float("nan"), "effect_r": float("nan")}
    if nz.size == 0:
        return out
    ranks = stats.rankdata(np.abs(nz))
    w_pos, w_neg = float(ranks[nz > 0].sum()), float(ranks[nz < 0].sum())
    out["effect_r"] = (w_pos - w_neg) / (w_pos + w_neg)
    res = stats.wilcoxon(nz)  # exact for small n, normal approximation for large n
    out["statistic"], out["p"] = float(res.statistic), float(res.pvalue)
    return out
