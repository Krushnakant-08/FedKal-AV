"""``TrajectoryBatch``: the data contract between the simulator and everything downstream.

Time convention (fixed for the whole project)::

    x[t]      true state at time t,           t = 0..T          shape (N, T+1, 4)
    u[t]      input applied during step t,    drives x[t]->x[t+1]   shape (N, T, 2)
    z[t]      GNSS measurement OF x[t+1],     t = 0..T-1        shape (N, T, 2)

So a filter loop is ``predict(x_hat, u_imu[t])`` followed by ``update(z[t])``.
``z`` is NaN exactly where ``z_mask`` is False. Filters must only use
``u_imu``, ``z``, ``z_mask`` and ``x0_est``; every other field is ground truth or
a diagnostic and is for losses, metrics and tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

import numpy as np

# field name -> (dtype, trailing-shape kind)
_SPEC: dict[str, np.dtype] = {
    "x": np.dtype(np.float64),
    "u_true": np.dtype(np.float64),
    "u_imu": np.dtype(np.float64),
    "z": np.dtype(np.float64),
    "z_mask": np.dtype(bool),
    "seg": np.dtype(np.int8),
    "x0_est": np.dtype(np.float64),
    "imu_bias": np.dtype(np.float64),
    "gnss_fault": np.dtype(bool),
    "outage": np.dtype(bool),
    "profile_idx": np.dtype(np.int8),
}


@dataclass
class TrajectoryBatch:
    """A batch of N trajectories of T steps. Arrays are treated as read-only by convention."""

    x: np.ndarray  # (N, T+1, 4) true state
    u_true: np.ndarray  # (N, T, 2) true [yaw_rate, accel]
    u_imu: np.ndarray  # (N, T, 2) IMU measurement of u_true
    z: np.ndarray  # (N, T, 2) GNSS position measurement of x[t+1]; NaN where unavailable
    z_mask: np.ndarray  # (N, T) bool, True = measurement available
    seg: np.ndarray  # (N, T) int8 segment id: 0 urban, 1 highway, 2 tunnel
    x0_est: np.ndarray  # (N, 4) noisy initial estimate, error ~ N(0, P0)
    imu_bias: np.ndarray  # (N, T, 2) diagnostic: true IMU bias (constant + random walk)
    gnss_fault: np.ndarray  # (N, T) bool diagnostic: injected GNSS spike at this sample
    outage: np.ndarray  # (N, T) bool diagnostic: GNSS outage (tunnel or window) at this step
    profile_idx: np.ndarray  # (N,) int8 index into SimConfig.profile_names
    dt: float
    model: str

    def __post_init__(self) -> None:
        self.validate()

    # ---------------------------------------------------------------- properties
    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def T(self) -> int:
        return int(self.u_true.shape[1])

    def __len__(self) -> int:
        return self.n

    # ---------------------------------------------------------------- validation
    def validate(self) -> None:
        n, steps = self.x.shape[0], self.u_true.shape[1]
        shapes = {
            "x": (n, steps + 1, 4),
            "u_true": (n, steps, 2),
            "u_imu": (n, steps, 2),
            "z": (n, steps, 2),
            "z_mask": (n, steps),
            "seg": (n, steps),
            "x0_est": (n, 4),
            "imu_bias": (n, steps, 2),
            "gnss_fault": (n, steps),
            "outage": (n, steps),
            "profile_idx": (n,),
        }
        for name, shape in shapes.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
            if arr.dtype != _SPEC[name]:
                raise ValueError(f"{name}: expected dtype {_SPEC[name]}, got {arr.dtype}")
        if not np.array_equal(np.isnan(self.z[..., 0]), ~self.z_mask):
            raise ValueError("z must be NaN exactly where z_mask is False")

    # ------------------------------------------------------------------ slicing
    def subset(self, idx: np.ndarray | list[int] | slice) -> "TrajectoryBatch":
        """Select trajectories (copy). ``idx`` may be a permutation, boolean mask or slice."""
        kw = {name: np.ascontiguousarray(getattr(self, name)[idx]) for name in _SPEC}
        return TrajectoryBatch(**kw, dt=self.dt, model=self.model)

    @classmethod
    def concat(cls, batches: list["TrajectoryBatch"]) -> "TrajectoryBatch":
        if not batches:
            raise ValueError("cannot concatenate zero batches")
        dt, model = batches[0].dt, batches[0].model
        if any(b.dt != dt or b.model != model or b.T != batches[0].T for b in batches):
            raise ValueError("batches must share dt, motion model and length")
        kw = {name: np.concatenate([getattr(b, name) for b in batches], axis=0) for name in _SPEC}
        return cls(**kw, dt=dt, model=model)

    # ------------------------------------------------------------------ hashing
    def content_hash(self) -> str:
        """SHA-256 over every array's name, dtype, shape and raw bytes (bit-exact identity)."""
        h = hashlib.sha256()
        for name in _SPEC:
            arr = np.ascontiguousarray(getattr(self, name))
            h.update(f"{name}|{arr.dtype}|{arr.shape}|".encode())
            h.update(arr.tobytes())
        h.update(f"dt={self.dt!r}|model={self.model}".encode())
        return h.hexdigest()

    def row_hashes(self) -> list[str]:
        """Per-trajectory SHA-256 (of the true state path), used to prove splits are disjoint."""
        return [hashlib.sha256(np.ascontiguousarray(self.x[i]).tobytes()).hexdigest() for i in range(self.n)]

    def __repr__(self) -> str:
        return f"TrajectoryBatch(n={self.n}, T={self.T}, dt={self.dt}, model='{self.model}')"


FIELD_NAMES: tuple[str, ...] = tuple(_SPEC)
assert set(FIELD_NAMES) <= {f.name for f in fields(TrajectoryBatch)}
