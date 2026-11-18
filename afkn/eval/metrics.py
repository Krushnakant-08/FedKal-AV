"""Metrics shared by the runner and the report (torch-free; duck-typed on ``x_hat``)."""

from __future__ import annotations

import numpy as np

from afkn.sim.batch import TrajectoryBatch


def position_errors(x_hat: np.ndarray, batch: TrajectoryBatch) -> np.ndarray:
    """Per-trajectory, per-step 2-D position error (N, T), time indices 1..T."""
    err = x_hat[:, 1:, :2] - batch.x[:, 1:, :2]
    return np.sqrt(np.sum(err**2, axis=-1))


def pooled_rmse(x_hat: np.ndarray, batch: TrajectoryBatch) -> float:
    """Pooled position RMSE (identical to ``filters.consistency.position_rmse``)."""
    e = position_errors(x_hat, batch)
    return float(np.sqrt(np.mean(e**2)))


def rmse_by_profile(x_hat: np.ndarray, batch: TrajectoryBatch, names: list[str]) -> dict[str, float]:
    """Pooled position RMSE per noise profile of the (mixed) batch."""
    e2 = position_errors(x_hat, batch) ** 2
    out: dict[str, float] = {}
    for idx, name in enumerate(names):
        sel = batch.profile_idx == idx
        if sel.any():
            out[name] = float(np.sqrt(np.mean(e2[sel])))
    return out


def rounds_to_converge(curve: list[tuple[int, float]], tol: float = 0.05) -> int | None:
    """First evaluated round whose RMSE is within ``tol`` of the final one (None if the curve is empty/non-finite)."""
    if not curve or not np.isfinite(curve[-1][1]):
        return None
    final = curve[-1][1]
    for r, v in curve:
        if np.isfinite(v) and v <= final * (1.0 + tol):
            return int(r)
    return int(curve[-1][0])
