"""Torch implementation of the server's trusted-set measurements."""

from __future__ import annotations

import dataclasses

import numpy as np

from afkn.models.config import KalmanNetConfig
from afkn.models.kalmannet import KalmanNet
from afkn.models.train import evaluate, train
from afkn.seeding import SeedTree
from afkn.sim.batch import TrajectoryBatch


class TorchOracle:
    """Reference update and RMSE on the trusted set, computed with the same trainer the clients use."""

    def __init__(self, kcfg: KalmanNetConfig, trusted: TrajectoryBatch, *, motion_model: str, dt: float,
                 ref_epochs: int, seed: int) -> None:
        self.kcfg = kcfg
        self.trusted = trusted
        self.ref_epochs = ref_epochs
        self.tree = SeedTree(seed).child("oracle")
        self.model = KalmanNet(kcfg, model=motion_model, dt=dt, seed=0)

    def reference_delta(self, global_params: np.ndarray, round_idx: int) -> np.ndarray:
        g = np.asarray(global_params, dtype=np.float32)
        self.model.set_flat_params(g)
        tcfg = dataclasses.replace(self.kcfg.train, epochs=self.ref_epochs)
        train(self.model, self.trusted, tcfg, seed=self.tree.child("ref", round_idx).int_seed())
        return (self.model.get_flat_params() - g).astype(np.float32)

    def rmse(self, params: np.ndarray) -> float:
        p = np.asarray(params, dtype=np.float32)
        if not np.isfinite(p).all():
            return float("nan")
        self.model.set_flat_params(p)
        return float(evaluate(self.model, self.trusted)["position_rmse"])
