"""Shared filter plumbing: result container, input checks, batched covariance helpers.

All filters are batched over trajectories (loop over time only) and follow the project
time convention: ``u_imu[t]`` drives the prediction ``x[t] -> x[t+1]`` and ``z[t]`` (if
``z_mask[t]``) is the GNSS measurement of ``x[t+1]``. Filters read only ``u_imu``, ``z``,
``z_mask`` and ``x0_est`` from the batch; ground truth is never touched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from afkn.filters.params import FilterParams
from afkn.sim.batch import TrajectoryBatch


@dataclass(frozen=True, eq=False)
class FilterResult:
    """Outputs of one filter run over a batch (N trajectories, T steps).

    ``x_hat``/``P`` are *posterior* and aligned with ``batch.x``: index 0 is the initial
    estimate ``x0_est`` with covariance ``P0``; index ``t+1`` is the estimate of ``x[t+1]``
    after processing step ``t``. ``x_pred``/``P_pred`` are the priors for ``x[t+1]``.
    ``innov``, ``S`` are NaN where no GNSS update happened.
    """

    x_hat: np.ndarray  # (N, T+1, 4)
    P: np.ndarray  # (N, T+1, 4, 4)
    x_pred: np.ndarray  # (N, T, 4)
    P_pred: np.ndarray  # (N, T, 4, 4)
    innov: np.ndarray  # (N, T, 2)
    S: np.ndarray  # (N, T, 2, 2)
    updated: np.ndarray  # (N, T) bool (== z_mask)
    bias_hat: np.ndarray | None = None  # (N, T+1, 2) estimated IMU bias (ESKF with bias only)

    @property
    def nis(self) -> np.ndarray:
        """Normalised innovation squared ``nu^T S^-1 nu``, shape (N, T), NaN where no update."""
        out = np.full(self.updated.shape, np.nan)
        if self.updated.any():
            nu, S = self.innov[self.updated], self.S[self.updated]
            out[self.updated] = np.einsum("ki,ki->k", nu, np.linalg.solve(S, nu[..., None])[..., 0])
        return out


def batch_cov(M: np.ndarray, n: int, name: str) -> np.ndarray:
    """Broadcast a shared (d,d) or per-trajectory (n,d,d) covariance to (n,d,d)."""
    M = np.asarray(M, dtype=float)
    if M.ndim == 2:
        return np.broadcast_to(M, (n,) + M.shape)
    if M.ndim == 3 and M.shape[0] == n:
        return M
    raise ValueError(f"{name}: expected (d,d) or ({n},d,d), got {M.shape}")


def check_inputs(params: FilterParams, batch: TrajectoryBatch) -> None:
    if not np.isclose(params.dt, batch.dt):
        raise ValueError(f"filter dt={params.dt} does not match batch dt={batch.dt}")


def symmetrize(P: np.ndarray) -> np.ndarray:
    return 0.5 * (P + np.swapaxes(P, -1, -2))


def solve_gain(PHt: np.ndarray, S: np.ndarray) -> np.ndarray:
    """K = PHt S^-1 for symmetric S, batched: PHt (n,d,2), S (n,2,2) -> (n,d,2)."""
    return np.swapaxes(np.linalg.solve(S, np.swapaxes(PHt, -1, -2)), -1, -2)


def joseph(P: np.ndarray, K: np.ndarray, Hm: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Joseph-form covariance update (stays symmetric PSD): (I-KH) P (I-KH)^T + K R K^T."""
    d = P.shape[-1]
    IKH = np.eye(d) - K @ Hm
    return symmetrize(IKH @ P @ np.swapaxes(IKH, -1, -2) + K @ R @ np.swapaxes(K, -1, -2))


class Filter(ABC):
    """Common interface: ``Filter(params).run(batch) -> FilterResult``."""

    name: str = "filter"

    def __init__(self, params: FilterParams) -> None:
        self.params = params

    @abstractmethod
    def run(self, batch: TrajectoryBatch, P0: np.ndarray | None = None) -> FilterResult:
        """Filter every trajectory in ``batch``. ``P0`` overrides ``params.P0`` if given."""
