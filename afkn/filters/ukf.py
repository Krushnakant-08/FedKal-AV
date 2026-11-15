"""Unscented Kalman filter (scaled unscented transform, additive process noise Q).

Predict uses an *augmented* sigma-point set over ``[x, n_u]`` (``n_u ~ N(0, Qu)`` is the IMU
white noise entering through the input), so the IMU noise is propagated through the true
nonlinearity instead of the Jacobian used by the EKF. Q is then added. The update uses a
fresh state-only sigma-point set through ``h`` (which is linear here, so the update equals
the Kalman update exactly; only the prediction differs from the EKF).
"""

from __future__ import annotations

import numpy as np

from afkn.filters.base import Filter, FilterResult, batch_cov, check_inputs, solve_gain, symmetrize
from afkn.filters.config import UKFSpec
from afkn.sim import motion
from afkn.sim.batch import TrajectoryBatch
from afkn.filters.params import FilterParams


def ut_weights(d: int, alpha: float, beta: float, kappa: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean weights, covariance weights and lambda of the scaled unscented transform."""
    lam = alpha**2 * (d + kappa) - d
    c = d + lam
    wm = np.full(2 * d + 1, 1.0 / (2.0 * c))
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return wm, wc, lam


def matrix_sqrt(P: np.ndarray) -> np.ndarray:
    """Batched square root ``L`` with ``L L^T = P``: Cholesky, falling back to eigh for PSD edge cases."""
    try:
        return np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(symmetrize(P))
        return V * np.sqrt(np.clip(w, 0.0, None))[..., None, :]


def sigma_points(
    mean: np.ndarray, L: np.ndarray, alpha: float, beta: float, kappa: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sigma points ``(n, 2d+1, d)`` for ``mean (n,d)`` and a square root ``L`` of the covariance."""
    d = mean.shape[-1]
    wm, wc, lam = ut_weights(d, alpha, beta, kappa)
    cols = np.sqrt(d + lam) * np.swapaxes(L, -1, -2)  # row i = i-th column of L
    X = np.concatenate([mean[:, None], mean[:, None] + cols, mean[:, None] - cols], axis=1)
    return X, wm, wc


class UKF(Filter):
    name = "ukf"

    def __init__(self, params: FilterParams, spec: UKFSpec | None = None) -> None:
        super().__init__(params)
        self.spec = spec or UKFSpec()

    def run(self, batch: TrajectoryBatch, P0: np.ndarray | None = None) -> FilterResult:
        p, s = self.params, self.spec
        check_inputs(p, batch)
        n, T = batch.n, batch.T
        R, Qu = batch_cov(p.R, n, "R"), batch_cov(p.Qu, n, "Qu")
        ut = (s.alpha, s.beta, s.kappa)

        x = batch.x0_est.copy()
        P = np.array(np.broadcast_to(p.P0 if P0 is None else P0, (n, 4, 4)))
        x_hat, P_hat = np.empty((n, T + 1, 4)), np.empty((n, T + 1, 4, 4))
        x_pred, P_pred = np.empty((n, T, 4)), np.empty((n, T, 4, 4))
        innov, S_all = np.full((n, T, 2), np.nan), np.full((n, T, 2, 2), np.nan)
        x_hat[:, 0], P_hat[:, 0] = x, P
        L_aug = np.zeros((n, 6, 6))
        L_aug[:, 4:, 4:] = matrix_sqrt(Qu)

        for t in range(T):
            u = batch.u_imu[:, t]
            # ---- predict: augmented sigma points over [x, input noise]
            L_aug[:, :4, :4] = matrix_sqrt(P)
            X, wm, wc = sigma_points(np.concatenate([x, np.zeros((n, 2))], axis=1), L_aug, *ut)
            Xp = motion.f(X[..., :4], u[:, None, :] + X[..., 4:], p.dt, p.model)
            xp = np.einsum("k,nkd->nd", wm, Xp)
            dX = Xp - xp[:, None]
            Pp = symmetrize(np.einsum("k,nki,nkj->nij", wc, dX, dX) + p.Q)

            # ---- update: state-only sigma points through h
            X2, wm2, wc2 = sigma_points(xp, matrix_sqrt(Pp), *ut)
            Z = motion.h(X2)
            zb = np.einsum("k,nkd->nd", wm2, Z)
            dZ = Z - zb[:, None]
            S = np.einsum("k,nki,nkj->nij", wc2, dZ, dZ) + R
            Pxz = np.einsum("k,nki,nkj->nij", wc2, X2 - xp[:, None], dZ)
            K = solve_gain(Pxz, S)
            m = batch.z_mask[:, t]
            z = np.where(m[:, None], batch.z[:, t], 0.0)
            nu = z - zb
            xu = xp + (K @ nu[..., None])[..., 0]
            Pu = symmetrize(Pp - K @ S @ np.swapaxes(K, -1, -2))

            x = np.where(m[:, None], xu, xp)
            P = np.where(m[:, None, None], Pu, Pp)
            x_pred[:, t], P_pred[:, t] = xp, Pp
            innov[:, t] = np.where(m[:, None], nu, np.nan)
            S_all[:, t] = np.where(m[:, None, None], S, np.nan)
            x_hat[:, t + 1], P_hat[:, t + 1] = x, P

        return FilterResult(x_hat, P_hat, x_pred, P_pred, innov, S_all, batch.z_mask.copy())
