"""KalmanNet-style filter: model-based predict step + GRU-based learned Kalman gain.

Per step ``t`` (``u[t]`` drives ``x[t] -> x[t+1]``; ``z[t]`` measures ``x[t+1]``)::

    x_pred = f(x_post, u_imu[t])                       known model (no learned dynamics)
    nu     = (z[t] - h(x_pred)) * available            innovation (0 when GNSS is missing)
    K      = GainNet(features)                         4x2 gain from a GRU (hidden size 64)
    x_post = x_pred + K @ nu                           update (x_post = x_pred when missing)

There is no covariance recursion: the network replaces the ``P``/``S`` computation of the EKF.
The model-based part uses the *known* ``f`` and ``h`` (the Jacobians of the EKF are not needed
because no covariance is propagated).

Features (all ``asinh(x / scale)`` squashed so tunnel-length dead-reckoning errors stay bounded;
index = position in the 15-vector):

    0-1    innovation                         z - h(x_{t|t-1})
    2-5    state-update difference            x_{t|t}   - x_{t|t-1}   (what the last update changed)
    6-9    forward-evolution difference       x_{t|t}   - x_{t-1|t-1} (posterior-to-posterior motion)
    10-13  forward-update difference          x_{t+1|t} - x_{t|t}     (motion the model predicts now)
    14     GNSS-available flag (raw 0/1)

ASSUMPTION (flagged in the PR): the Build Plan names four features but not their exact
definitions; the three state differences above are our reading, loosely following the KalmanNet
paper's F3/F4 differences. Change them in one place (``KalmanNet.step``) if the thesis defines them differently.

Parameters are exposed as one flat ``float32`` vector (``get_flat_params`` / ``set_flat_params``)
for the federated loop and exact communication accounting (``num_params * 4`` bytes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from torch import nn

from afkn.models.config import N_FEATURES, N_GAIN, KalmanNetConfig, KalmanNetSpec
from afkn.models.motion_torch import MODELS, f_torch, h_torch
from afkn.sim.batch import TrajectoryBatch


@dataclass
class KNState:
    """Recurrent state carried between steps (and between truncated-BPTT chunks)."""

    x_post: torch.Tensor  # (N, 4)  x_{t|t}
    x_post_prev: torch.Tensor  # (N, 4)  x_{t-1|t-1}
    x_prior: torch.Tensor  # (N, 4)  x_{t|t-1}
    hidden: torch.Tensor  # (N, hidden)

    def detach(self) -> KNState:
        return KNState(self.x_post.detach(), self.x_post_prev.detach(), self.x_prior.detach(), self.hidden.detach())


class StepOut(NamedTuple):
    state: KNState
    x_pred: torch.Tensor  # (N, 4)
    gain: torch.Tensor  # (N, 4, 2)
    innov: torch.Tensor  # (N, 2)


@dataclass(frozen=True, eq=False)
class KalmanNetResult:
    """Inference output, aligned with ``batch.x`` (index 0 = ``x0_est``). float64 NumPy."""

    x_hat: np.ndarray  # (N, T+1, 4)
    x_pred: np.ndarray  # (N, T, 4)
    gain: np.ndarray  # (N, T, 4, 2)
    innov: np.ndarray  # (N, T, 2) (0 where GNSS unavailable)
    updated: np.ndarray  # (N, T) bool


class KalmanNet(nn.Module):
    """GRU-based Kalman-gain network around the known motion model."""

    def __init__(
        self,
        config: KalmanNetConfig | None = None,
        *,
        model: str = "ctrv",
        dt: float = 0.1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if model not in MODELS:
            raise ValueError(f"unknown motion model '{model}', expected one of {MODELS}")
        self.config = config or KalmanNetConfig()
        spec = self.config.net
        self.spec: KalmanNetSpec = spec
        self.model_name = model
        self.dt = float(dt)

        self.fc_in = nn.Linear(N_FEATURES, spec.hidden)
        self.gru = nn.GRUCell(spec.hidden, spec.hidden)
        self.fc1 = nn.Linear(spec.hidden, spec.fc_hidden)
        self.fc2 = nn.Linear(spec.fc_hidden, N_GAIN)

        # Fixed (non-trainable, non-persistent) scaling constants.
        self.register_buffer("state_scale", torch.tensor(spec.state_scale, dtype=torch.float32), persistent=False)
        self.register_buffer("gain_scale", torch.tensor(spec.gain_scale, dtype=torch.float32), persistent=False)
        self.reset_parameters(seed)

    # ------------------------------------------------------------------ parameters
    def reset_parameters(self, seed: int) -> None:
        """Deterministic initialisation from a local generator (the global torch RNG is never used)."""
        spec = self.spec
        g = torch.Generator().manual_seed(int(seed))

        def fill(t: torch.Tensor, bound: float) -> None:
            t.copy_((torch.rand(t.shape, generator=g) * 2.0 - 1.0) * bound)

        with torch.no_grad():
            fill(self.fc_in.weight, 1.0 / math.sqrt(N_FEATURES))
            fill(self.fc_in.bias, 1.0 / math.sqrt(N_FEATURES))
            for p in (self.gru.weight_ih, self.gru.weight_hh, self.gru.bias_ih, self.gru.bias_hh):
                fill(p, 1.0 / math.sqrt(spec.hidden))
            fill(self.fc1.weight, 1.0 / math.sqrt(spec.hidden))
            fill(self.fc1.bias, 1.0 / math.sqrt(spec.hidden))
            fill(self.fc2.weight, spec.out_init_scale / math.sqrt(spec.fc_hidden))
            self.fc2.bias.zero_()
            if spec.init_pos_gain != 0.0:  # K[0,0] and K[1,1] in the row-major 4x2 layout
                self.fc2.bias[0] = spec.init_pos_gain / spec.gain_scale[0]
                self.fc2.bias[3] = spec.init_pos_gain / spec.gain_scale[1]

    @property
    def num_params(self) -> int:
        """Number of trainable parameters (communication cost is ``num_params * 4`` bytes)."""
        return int(sum(p.numel() for p in self.parameters()))

    def param_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Names and shapes in flat-vector order."""
        return [(n, tuple(p.shape)) for n, p in self.named_parameters()]

    def get_flat_params(self) -> np.ndarray:
        """All parameters as one contiguous 1-D ``float32`` copy (safe to mutate)."""
        with torch.no_grad():
            parts = [p.detach().cpu().numpy().astype(np.float32).ravel() for p in self.parameters()]
        return np.ascontiguousarray(np.concatenate(parts))

    def set_flat_params(self, flat: np.ndarray) -> None:
        """Load a flat vector produced by ``get_flat_params``. Rejects wrong size, rank, NaN and inf."""
        arr = np.asarray(flat)
        if arr.ndim != 1:
            raise ValueError(f"flat params must be 1-D, got shape {arr.shape}")
        if arr.size != self.num_params:
            raise ValueError(f"expected {self.num_params} parameters, got {arr.size}")
        if not np.isfinite(arr).all():
            raise ValueError("flat params contain NaN or inf")
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for p in self.parameters():
                n = p.numel()
                p.copy_(torch.from_numpy(arr[offset : offset + n].copy()).view(p.shape))
                offset += n

    # --------------------------------------------------------------------- dynamics
    def init_state(self, x0_est: torch.Tensor) -> KNState:
        x0 = x0_est.to(torch.float32)
        return KNState(x0, x0, x0, torch.zeros(x0.shape[0], self.spec.hidden, dtype=torch.float32))

    def step(self, state: KNState, u: torch.Tensor, z: torch.Tensor, avail: torch.Tensor) -> StepOut:
        """One filter step. ``u (N,2)``, ``z (N,2)`` (any finite value where unavailable), ``avail (N,)`` in {0,1}."""
        x_pred = f_torch(state.x_post, u, self.dt, self.model_name)
        innov = (z - h_torch(x_pred)) * avail[:, None]
        sq = self.state_scale
        feats = torch.cat(
            [
                torch.asinh(innov / self.spec.innov_scale),
                torch.asinh((state.x_post - state.x_prior) / sq),
                torch.asinh((state.x_post - state.x_post_prev) / sq),
                torch.asinh((x_pred - state.x_post) / sq),
                avail[:, None],
            ],
            dim=1,
        )
        hidden = self.gru(torch.relu(self.fc_in(feats)), state.hidden)
        gain = self.fc2(torch.relu(self.fc1(hidden))).view(-1, 4, 2) * self.gain_scale[None, :, None]
        x_post = x_pred + torch.bmm(gain, innov.unsqueeze(-1)).squeeze(-1)  # innov == 0 when unavailable
        return StepOut(KNState(x_post, state.x_post, x_pred, hidden), x_pred, gain, innov)

    def forward(
        self,
        u: torch.Tensor,
        z: torch.Tensor,
        avail: torch.Tensor,
        x0_est: torch.Tensor | None = None,
        state: KNState | None = None,
        return_aux: bool = False,
    ):
        """Run over a sequence. ``u, z: (N, T, 2)``, ``avail: (N, T)``.

        Returns ``(x_post (N, T, 4), final_state)`` or, with ``return_aux``,
        ``(x_post, final_state, x_pred (N,T,4), gain (N,T,4,2), innov (N,T,2))``.
        Pass the returned (detached) state back in to continue a sequence in chunks (truncated BPTT).
        """
        if state is None:
            if x0_est is None:
                raise ValueError("either x0_est or state must be given")
            state = self.init_state(x0_est)
        xs, preds, gains, innovs = [], [], [], []
        for t in range(u.shape[1]):
            out = self.step(state, u[:, t], z[:, t], avail[:, t])
            state = out.state
            xs.append(state.x_post)
            if return_aux:
                preds.append(out.x_pred)
                gains.append(out.gain)
                innovs.append(out.innov)
        x_post = torch.stack(xs, dim=1)
        if return_aux:
            return x_post, state, torch.stack(preds, 1), torch.stack(gains, 1), torch.stack(innovs, 1)
        return x_post, state

    # -------------------------------------------------------------------- inference
    @torch.no_grad()
    def run(self, batch: TrajectoryBatch) -> KalmanNetResult:
        """Filter a NumPy ``TrajectoryBatch``. Reads only ``u_imu``, ``z``, ``z_mask`` and ``x0_est``."""
        if not math.isclose(batch.dt, self.dt):
            raise ValueError(f"model dt={self.dt} does not match batch dt={batch.dt}")
        was_training = self.training
        self.eval()
        try:
            u = torch.as_tensor(batch.u_imu, dtype=torch.float32)
            z = torch.as_tensor(np.nan_to_num(batch.z, nan=0.0), dtype=torch.float32)
            avail = torch.as_tensor(batch.z_mask, dtype=torch.float32)
            x0 = torch.as_tensor(batch.x0_est, dtype=torch.float32)
            x_post, _, x_pred, gain, innov = self.forward(u, z, avail, x0_est=x0, return_aux=True)
        finally:
            self.train(was_training)
        x_hat = np.concatenate([np.asarray(batch.x0_est, dtype=np.float64)[:, None], x_post.double().numpy()], axis=1)
        return KalmanNetResult(
            x_hat=x_hat,
            x_pred=x_pred.double().numpy(),
            gain=gain.double().numpy(),
            innov=innov.double().numpy(),
            updated=batch.z_mask.copy(),
        )
