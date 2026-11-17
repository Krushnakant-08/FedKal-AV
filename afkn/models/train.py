"""Supervised training of KalmanNet: MSE on the state, Adam, gradient clipping, truncated BPTT.

Training sees ground-truth states only through the loss; the network inputs are exactly what a
filter may use (``u_imu``, ``z``, ``z_mask``, ``x0_est``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from afkn.filters.consistency import position_rmse
from afkn.models.config import TrainSpec
from afkn.models.kalmannet import KalmanNet, KalmanNetResult
from afkn.models.motion_torch import wrap_angle_torch
from afkn.seeding import SeedTree
from afkn.sim.batch import TrajectoryBatch


@dataclass
class TrainLog:
    """Per-optimiser-step losses (one step per truncated-BPTT chunk) and gradient norms (before clipping)."""

    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)


def mse_state_loss(x_hat: torch.Tensor, x_true: torch.Tensor, weights: tuple[float, float, float, float]) -> torch.Tensor:
    """Weighted MSE on ``[px, py, heading, speed]`` with the heading error wrapped to (-pi, pi]."""
    err = x_hat - x_true
    err = torch.cat([err[..., :2], wrap_angle_torch(err[..., 2:3]), err[..., 3:]], dim=-1)
    w = torch.as_tensor(weights, dtype=err.dtype)
    return (err.pow(2) * w).mean()


def _tensors(batch: TrajectoryBatch) -> dict[str, torch.Tensor]:
    f32 = torch.float32
    return {
        "u": torch.as_tensor(batch.u_imu, dtype=f32),
        "z": torch.as_tensor(np.nan_to_num(batch.z, nan=0.0), dtype=f32),
        "avail": torch.as_tensor(batch.z_mask, dtype=f32),
        "x0": torch.as_tensor(batch.x0_est, dtype=f32),
        "x": torch.as_tensor(batch.x, dtype=f32),
    }


def train(
    model: KalmanNet,
    batch: TrajectoryBatch,
    train_cfg: TrainSpec | None = None,
    *,
    seed: int = 0,
) -> TrainLog:
    """Train ``model`` in place on ``batch``. Deterministic given ``seed`` (batch order from a SeedTree stream)."""
    cfg = train_cfg or model.config.train
    data = _tensors(batch)
    n, steps = batch.n, batch.T
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = SeedTree(seed).child("kalmannet", "train").rng()
    log = TrainLog()

    model.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        epoch_losses: list[float] = []
        for s in range(0, n, cfg.batch_size):
            idx = torch.as_tensor(order[s : s + cfg.batch_size], dtype=torch.long)
            u, z, avail, x_true = data["u"][idx], data["z"][idx], data["avail"][idx], data["x"][idx]
            state = model.init_state(data["x0"][idx])
            for c0 in range(0, steps, cfg.chunk_len):
                c1 = min(c0 + cfg.chunk_len, steps)
                x_hat, state = model(u[:, c0:c1], z[:, c0:c1], avail[:, c0:c1], state=state)
                loss = mse_state_loss(x_hat, x_true[:, c0 + 1 : c1 + 1], cfg.loss_weights)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                state = state.detach()  # truncate BPTT at the chunk boundary
                log.losses.append(float(loss.item()))
                log.grad_norms.append(float(gnorm))
                epoch_losses.append(log.losses[-1])
        log.epoch_losses.append(float(np.mean(epoch_losses)) if epoch_losses else float("nan"))
    model.eval()
    return log


def evaluate(model: KalmanNet, batch: TrajectoryBatch) -> dict[str, float]:
    """Pooled 2-D position RMSE (same definition as the filter baselines) and state MSE."""
    res: KalmanNetResult = model.run(batch)
    err = res.x_hat[:, 1:] - batch.x[:, 1:]
    err[..., 2] = (err[..., 2] + np.pi) % (2.0 * np.pi) - np.pi
    return {"position_rmse": position_rmse(res, batch), "state_mse": float(np.mean(err**2))}  # type: ignore[arg-type]
