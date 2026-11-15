"""Filter evaluation: NEES / NIS, chi-square bounds, position RMSE.

Definitions (all computed from data, nothing assumed):
    NEES_t = e_t^T P_t^-1 e_t with e = x_hat - x_true (heading difference wrapped)
    ANEES at time t = mean over the N independent trajectories of NEES_t. For a consistent
        filter N * ANEES_t ~ chi2(N * n_x), which gives the two-sided bounds below.
    NIS = nu^T S^-1 nu over available GNSS updates ~ chi2(2) for a consistent filter; for a
        correct filter the innovations are white, so pooled NIS samples are ~independent.
    position RMSE = sqrt(mean ||p_hat - p_true||^2) (2-D Euclidean error).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from afkn.filters.base import FilterResult
from afkn.sim.batch import TrajectoryBatch
from afkn.sim.motion import wrap_angle


def state_error(x_hat: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """``x_hat - x_true`` with the heading component wrapped to (-pi, pi]."""
    e = np.array(x_hat - x_true)
    e[..., 2] = wrap_angle(e[..., 2])
    return e


def nees(result: FilterResult, batch: TrajectoryBatch) -> np.ndarray:
    """NEES for every trajectory and time index, shape (N, T+1) (index 0 is the initial estimate)."""
    e = state_error(result.x_hat, batch.x)
    return np.einsum("nti,nti->nt", e, np.linalg.solve(result.P, e[..., None])[..., 0])


def anees_bounds(n_runs: int, dim: int = 4, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided ``conf`` bounds for the across-trajectory average NEES of a consistent filter."""
    a = (1.0 - conf) / 2.0
    return (float(chi2.ppf(a, n_runs * dim) / n_runs), float(chi2.ppf(1.0 - a, n_runs * dim) / n_runs))


def position_rmse(result: FilterResult, batch: TrajectoryBatch) -> float:
    """Pooled 2-D position RMSE over all trajectories and time indices 1..T."""
    err = result.x_hat[:, 1:, :2] - batch.x[:, 1:, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=-1))))


def consistency_summary(result: FilterResult, batch: TrajectoryBatch, conf: float = 0.95, burn_in: int = 0) -> dict:
    """Numbers used by tests and (later) the evaluation scripts. ``burn_in`` skips initial time indices."""
    nees_all = nees(result, batch)[:, burn_in:]
    anees_t = nees_all.mean(axis=0)
    lo, hi = anees_bounds(batch.n, 4, conf)
    nis = result.nis[np.isfinite(result.nis)]
    return {
        "n_trajectories": batch.n,
        "anees_mean": float(anees_t.mean()),
        "anees_bounds": (lo, hi),
        "anees_frac_in_bounds": float(np.mean((anees_t >= lo) & (anees_t <= hi))),
        "nis_mean": float(nis.mean()) if nis.size else float("nan"),
        "nis_coverage": float(np.mean(nis <= chi2.ppf(conf, 2))) if nis.size else float("nan"),
        "n_nis": int(nis.size),
        "position_rmse": position_rmse(result, batch),
    }
