"""Federated client: local KalmanNet training from a broadcast, returning a byte-level update."""

from __future__ import annotations

import dataclasses

import numpy as np

from afkn.data.partition import ClientData
from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.models.config import KalmanNetConfig
from afkn.models.kalmannet import KalmanNet
from afkn.models.train import evaluate, train


class Client:
    """One honest client. Owns its data and a model instance; stateless between rounds."""

    def __init__(self, data: ClientData, kcfg: KalmanNetConfig, *, motion_model: str, dt: float) -> None:
        self.data = data
        self.kcfg = kcfg
        self.model = KalmanNet(kcfg, model=motion_model, dt=dt, seed=0)

    @property
    def client_id(self) -> int:
        return self.data.client_id

    def local_update(self, bc: GlobalBroadcast, local_epochs: int, seed: int) -> ClientUpdate:
        """Train ``local_epochs`` from the broadcast model on the client's train split.

        Reported ``signals`` (what Layer 2's claim check will later compare with the server's own
        measurement): ``val_rmse_before`` / ``val_rmse_after`` (position RMSE on the client's val split
        before/after local training), ``innov_ms`` (mean squared GNSS innovation of the updated model on val;
        KalmanNet has no covariance, so this is a NIS proxy without normalisation) and ``train_loss``.
        """
        self.model.set_flat_params(bc.params)
        val = self.data.val
        before = evaluate(self.model, val)["position_rmse"]
        tcfg = dataclasses.replace(self.kcfg.train, epochs=local_epochs)
        log = train(self.model, self.data.train, tcfg, seed=seed)
        res = self.model.run(val)
        upd_mask = res.updated
        innov_ms = float(np.mean(np.sum(res.innov[upd_mask] ** 2, axis=-1))) if upd_mask.any() else 0.0
        after = evaluate(self.model, val)["position_rmse"]
        delta = (self.model.get_flat_params() - bc.params.astype(np.float32)).astype(np.float32)
        signals = {
            "val_rmse_before": before,
            "val_rmse_after": after,
            "innov_ms": innov_ms,
            "train_loss": log.epoch_losses[-1] if log.epoch_losses else float("nan"),
        }
        return ClientUpdate(
            client_id=self.client_id,
            round=bc.round,
            nonce=bc.nonce,
            model_hash=bc.model_hash,
            delta=delta,
            n_samples=self.data.n_train,
            signals=signals,
        )
