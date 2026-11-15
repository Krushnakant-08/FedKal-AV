"""Extended Kalman filter on the shared motion model.

Predict:  x- = f(x, u_imu),  P- = F P F^T + Q + G Qu G^T
Update:   nu = z - h(x-),  S = H P- H^T + R,  K = P- H^T S^-1,  Joseph-form covariance.
When ``z_mask`` is False the update is skipped (posterior = prior): outages are dead reckoning.
"""

from __future__ import annotations

import numpy as np

from afkn.filters.base import Filter, FilterResult, batch_cov, check_inputs, joseph, solve_gain, symmetrize
from afkn.sim import motion
from afkn.sim.batch import TrajectoryBatch


class EKF(Filter):
    name = "ekf"

    def run(self, batch: TrajectoryBatch, P0: np.ndarray | None = None) -> FilterResult:
        p = self.params
        check_inputs(p, batch)
        n, T = batch.n, batch.T
        R, Qu = batch_cov(p.R, n, "R"), batch_cov(p.Qu, n, "Qu")
        Hm = motion.H

        x = batch.x0_est.copy()
        P = np.array(np.broadcast_to(p.P0 if P0 is None else P0, (n, 4, 4)))
        x_hat, P_hat = np.empty((n, T + 1, 4)), np.empty((n, T + 1, 4, 4))
        x_pred, P_pred = np.empty((n, T, 4)), np.empty((n, T, 4, 4))
        innov, S_all = np.full((n, T, 2), np.nan), np.full((n, T, 2, 2), np.nan)
        x_hat[:, 0], P_hat[:, 0] = x, P

        for t in range(T):
            u = batch.u_imu[:, t]
            F = motion.jac_x(x, u, p.dt, p.model)
            G = motion.jac_u(x, u, p.dt, p.model)
            xp = motion.f(x, u, p.dt, p.model)
            Pp = symmetrize(F @ P @ np.swapaxes(F, -1, -2) + p.Q + G @ Qu @ np.swapaxes(G, -1, -2))

            m = batch.z_mask[:, t]
            z = np.where(m[:, None], batch.z[:, t], 0.0)
            nu = z - motion.h(xp)
            PHt = Pp @ Hm.T
            S = Hm @ PHt + R
            K = solve_gain(PHt, S)
            xu = xp + (K @ nu[..., None])[..., 0]
            Pu = joseph(Pp, K, Hm, R)

            x = np.where(m[:, None], xu, xp)
            P = np.where(m[:, None, None], Pu, Pp)
            x_pred[:, t], P_pred[:, t] = xp, Pp
            innov[:, t] = np.where(m[:, None], nu, np.nan)
            S_all[:, t] = np.where(m[:, None, None], S, np.nan)
            x_hat[:, t + 1], P_hat[:, t + 1] = x, P

        return FilterResult(x_hat, P_hat, x_pred, P_pred, innov, S_all, batch.z_mask.copy())
