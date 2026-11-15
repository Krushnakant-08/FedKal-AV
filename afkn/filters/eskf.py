"""SIMPLIFIED error-state Kalman filter for GNSS/IMU. Read what is simplified before using it.

What it is
    A nominal state ``x_nom = [px, py, heading, speed]`` driven by the (bias-corrected) IMU, plus
    an error state ``dx = [dpx, dpy, dheading, dspeed, d_bias_yaw, d_bias_acc]`` whose covariance is
    propagated with the linearised error dynamics. Each GNSS update estimates ``dx``, injects it
    into the nominal state and the bias estimate, and resets the error to zero.

What is simplified (this is NOT a full INS error-state filter)
    * planar 4-D kinematics: no 3-D attitude, no quaternions, no gravity, no earth rate;
    * the IMU "measurements" are directly yaw rate and longitudinal acceleration, with a 2-D
      additive bias [yaw-rate bias, accel bias] modelled as a random walk; scale error is ignored;
    * no lever arm, no GNSS time delay, no bias-observability analysis;
    * the error-reset Jacobian is the identity (valid because the heading error is a plain
      scalar here, not a rotation vector).
    With ``estimate_bias = False`` the filter reduces *exactly* to the EKF (tested).

Error dynamics (``u_c = u_imu - b_hat``, true ``u = u_c - d_b - n``)
    dx' = F dx - G d_b - G n + w,        d_b' = d_b + w_b
"""

from __future__ import annotations

import numpy as np

from afkn.filters.base import Filter, FilterResult, batch_cov, check_inputs, joseph, solve_gain, symmetrize
from afkn.filters.config import ESKFSpec
from afkn.filters.params import FilterParams
from afkn.sim import motion
from afkn.sim.batch import TrajectoryBatch


class ESKF(Filter):
    name = "eskf_simplified"

    def __init__(self, params: FilterParams, spec: ESKFSpec | None = None) -> None:
        super().__init__(params)
        self.spec = spec or ESKFSpec()

    def run(self, batch: TrajectoryBatch, P0: np.ndarray | None = None) -> FilterResult:
        p, s = self.params, self.spec
        check_inputs(p, batch)
        n, T = batch.n, batch.T
        nb = 2 if s.estimate_bias else 0
        d = 4 + nb
        R, Qu = batch_cov(p.R, n, "R"), batch_cov(p.Qu, n, "Qu")
        Hm = np.zeros((2, d))
        Hm[:, :2] = np.eye(2)

        x = batch.x0_est.copy()
        b = np.zeros((n, 2))
        P = np.zeros((n, d, d))
        P[:, :4, :4] = p.P0 if P0 is None else P0
        Qb = np.diag(np.square(s.bias_rw_std)) * p.dt
        if nb:
            P[:, 4:, 4:] = np.diag(np.square(s.bias_init_std))

        x_hat, P_hat = np.empty((n, T + 1, 4)), np.empty((n, T + 1, 4, 4))
        x_pred, P_pred = np.empty((n, T, 4)), np.empty((n, T, 4, 4))
        innov, S_all = np.full((n, T, 2), np.nan), np.full((n, T, 2, 2), np.nan)
        b_hat = np.zeros((n, T + 1, 2)) if nb else None
        x_hat[:, 0], P_hat[:, 0] = x, P[:, :4, :4]

        for t in range(T):
            uc = batch.u_imu[:, t] - b  # bias-corrected input
            F = motion.jac_x(x, uc, p.dt, p.model)
            G = motion.jac_u(x, uc, p.dt, p.model)
            xp = motion.f(x, uc, p.dt, p.model)

            A = np.zeros((n, d, d))
            A[:, :4, :4] = F
            Qe = np.zeros((n, d, d))
            Qe[:, :4, :4] = p.Q + G @ Qu @ np.swapaxes(G, -1, -2)
            if nb:
                A[:, :4, 4:] = -G
                A[:, 4:, 4:] = np.eye(2)
                Qe[:, 4:, 4:] = Qb
            Pp = symmetrize(A @ P @ np.swapaxes(A, -1, -2) + Qe)

            m = batch.z_mask[:, t]
            z = np.where(m[:, None], batch.z[:, t], 0.0)
            nu = z - motion.h(xp)
            PHt = Pp @ Hm.T
            S = Hm @ PHt + R
            K = solve_gain(PHt, S)
            dx = (K @ nu[..., None])[..., 0]  # estimated error state
            Pu = joseph(Pp, K, Hm, R)

            # inject the error into the nominal state, then reset the error (identity reset Jacobian)
            x = np.where(m[:, None], xp + dx[:, :4], xp)
            if nb:
                b = np.where(m[:, None], b + dx[:, 4:], b)
            P = np.where(m[:, None, None], Pu, Pp)
            x_pred[:, t], P_pred[:, t] = xp, Pp[:, :4, :4]
            innov[:, t] = np.where(m[:, None], nu, np.nan)
            S_all[:, t] = np.where(m[:, None, None], S, np.nan)
            x_hat[:, t + 1], P_hat[:, t + 1] = x, P[:, :4, :4]
            if nb:
                b_hat[:, t + 1] = b

        return FilterResult(x_hat, P_hat, x_pred, P_pred, innov, S_all, batch.z_mask.copy(), bias_hat=b_hat)
