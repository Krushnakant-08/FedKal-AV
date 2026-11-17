"""Layer 2 score primitives (torch-free, pure NumPy)."""

from __future__ import annotations

import numpy as np


def clip_to_median(deltas: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Scale each row so its norm is at most ``c x median(norm)``.

    Returns ``(clipped, factors, median_norm)``; ``factors`` in (0, 1]. If the median norm is 0 nothing is clipped.
    """
    d = np.asarray(deltas, dtype=np.float64)
    norms = np.linalg.norm(d, axis=1)
    med = float(np.median(norms))
    factors = np.ones(len(d))
    if med > 0:
        over = norms > c * med
        factors[over] = (c * med) / norms[over]
    return d * factors[:, None], factors, med


def direction_scores(deltas: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """FLTrust-style ``max(0, cos(delta_i, ref))``; a zero-norm delta or reference scores 0."""
    d = np.asarray(deltas, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    nr = np.linalg.norm(r)
    nd = np.linalg.norm(d, axis=1)
    out = np.zeros(len(d))
    ok = (nd > 0) & (nr > 0)
    out[ok] = np.maximum(0.0, (d[ok] @ r) / (nd[ok] * nr))
    return out


def perf_scores(gains: np.ndarray, scale: float) -> np.ndarray:
    """``sigmoid(gain / scale)``; NaN gain (non-finite measured RMSE) scores 0."""
    g = np.asarray(gains, dtype=np.float64)
    z = np.where(np.isnan(g), -np.inf, g / scale)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def claim_flags(reported_gain: np.ndarray, measured_gain: np.ndarray, reported_min: float, measured_max: float) -> np.ndarray:
    """"Liar" = claims an improvement above ``reported_min`` while the server measures a gain below ``measured_max``.

    A non-finite reported gain (missing or NaN signals) cannot be verified and is flagged too; a non-finite
    measured gain counts as measured degradation.
    """
    rep = np.asarray(reported_gain, dtype=np.float64)
    mea = np.asarray(measured_gain, dtype=np.float64)
    unverifiable = ~np.isfinite(rep)
    claims = np.where(np.isfinite(rep), rep > reported_min, False)
    degraded = np.where(np.isfinite(mea), mea < measured_max, True)
    return unverifiable | (claims & degraded)


def combine_scores(s_dir: np.ndarray, s_perf: np.ndarray, how: str) -> np.ndarray:
    if how == "product":
        return np.asarray(s_dir) * np.asarray(s_perf)
    if how == "mean":
        return 0.5 * (np.asarray(s_dir) + np.asarray(s_perf))
    raise ValueError(f"unknown combine '{how}'")
