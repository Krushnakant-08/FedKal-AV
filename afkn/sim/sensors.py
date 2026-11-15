"""Sensor models: per-device parameter realisation, IMU and GNSS measurement generation.

Per-device constants (biases, scale error, fault direction) are drawn ONCE per
client via :func:`realize_device` and shared by all of that client's
trajectories; per-sample noise, the IMU random walk, dropouts and outage windows
are drawn fresh for every trajectory from named child streams of a
:class:`~afkn.seeding.SeedTree`.

Every random stream draws a fixed number of variates regardless of the data it
ends up masking, so changing e.g. ``p_drop`` never shifts unrelated noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from afkn.seeding import SeedTree
from afkn.sim.config import ProfileSpec

TUNNEL_SEG_ID = 2  # index of "tunnel" in config.SEGMENT_TYPES


@dataclass(frozen=True)
class DeviceParams:
    """Realised sensor parameters of one device (what the config's *effective* sigmas are)."""

    profile: str
    gnss_sigma: float
    gnss_bias: tuple[float, float]  # constant offset added to every GNSS sample [m]
    p_drop: float
    outage_windows: tuple[int, int]
    outage_len: tuple[int, int]
    imu_sigma: tuple[float, float]  # (yaw rate, accel) white-noise std
    imu_bias0: tuple[float, float]  # constant bias at the start of each trajectory
    imu_rw: tuple[float, float]  # random-walk std per sqrt(second)
    imu_scale: tuple[float, float]  # multiplicative scale error, u_meas = (1+scale) u + ...
    fault_mode: str | None
    spike_prob: float  # effective: 0 unless fault_mode == "random_spikes"
    spike_std: float

    def R(self) -> np.ndarray:
        """GNSS measurement-noise covariance the *honest* filter would assume (2x2)."""
        return self.gnss_sigma**2 * np.eye(2)

    def to_dict(self) -> dict:
        return asdict(self)


def realize_device(
    profile_name: str,
    profile: ProfileSpec,
    rng: np.random.Generator,
    *,
    sigma_scale: float = 1.0,
    fault_mode: str | None = None,
) -> DeviceParams:
    """Draw the per-device constants. ``sigma_scale`` scales GNSS sigma, IMU white noise and IMU random walk."""
    mode = fault_mode if fault_mode is not None else profile.fault.mode
    # Fixed draw count/order, independent of the configuration.
    g_bias = rng.standard_normal(2)
    fault_angle = rng.uniform(0.0, 2.0 * np.pi)
    fault_sign = 1.0 if rng.random() < 0.5 else -1.0
    b0 = rng.standard_normal(2)
    scale = rng.standard_normal(2)

    g, im, fl = profile.gnss, profile.imu, profile.fault
    gnss_bias = g.bias_std * g_bias
    imu_bias0 = np.array([im.bias0_std_yaw, im.bias0_std_acc]) * b0
    if mode == "constant_bias":
        gnss_bias = gnss_bias + fl.gnss_bias_m * np.array([np.cos(fault_angle), np.sin(fault_angle)])
        imu_bias0[0] += fault_sign * fl.imu_yaw_bias

    return DeviceParams(
        profile=profile_name,
        gnss_sigma=float(g.sigma * sigma_scale),
        gnss_bias=(float(gnss_bias[0]), float(gnss_bias[1])),
        p_drop=float(g.p_drop),
        outage_windows=g.outage_windows,
        outage_len=g.outage_len,
        imu_sigma=(float(im.sigma_yaw * sigma_scale), float(im.sigma_acc * sigma_scale)),
        imu_bias0=(float(imu_bias0[0]), float(imu_bias0[1])),
        imu_rw=(float(im.rw_yaw * sigma_scale), float(im.rw_acc * sigma_scale)),
        imu_scale=(float(im.scale_std * scale[0]), float(im.scale_std * scale[1])),
        fault_mode=mode,
        spike_prob=float(fl.spike_prob) if mode == "random_spikes" else 0.0,
        spike_std=float(fl.spike_std),
    )


def apply_imu(
    u_true: np.ndarray, dev: DeviceParams, dt: float, tree: SeedTree
) -> tuple[np.ndarray, np.ndarray]:
    """IMU measurement of ``u_true`` (N, T, 2). Returns ``(u_imu, bias)`` with ``bias`` (N, T, 2).

    ``u_imu = (1 + scale) * u_true + bias(t) + N(0, sigma^2)`` with
    ``bias(t) = bias0 + cumsum(rw * sqrt(dt) * xi)``.
    """
    n, steps, _ = u_true.shape
    white = tree.child("white").rng().standard_normal((n, steps, 2))
    walk = tree.child("walk").rng().standard_normal((n, steps, 2))
    bias = np.asarray(dev.imu_bias0) + np.cumsum(walk * (np.sqrt(dt) * np.asarray(dev.imu_rw)), axis=1)
    u_imu = u_true * (1.0 + np.asarray(dev.imu_scale)) + bias + white * np.asarray(dev.imu_sigma)
    return u_imu, bias


def apply_gnss(
    pos_next: np.ndarray, seg: np.ndarray, dev: DeviceParams, tree: SeedTree
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """GNSS measurement of ``pos_next`` = x[:, 1:, :2] (N, T, 2).

    Returns ``(z, z_mask, outage, spike)``. ``z`` is NaN where ``z_mask`` is False.
    Unavailable = i.i.d. dropout OR outage; outage = tunnel segment OR a random window.
    """
    n, steps, _ = pos_next.shape
    noise = tree.child("noise").rng().standard_normal((n, steps, 2)) * dev.gnss_sigma
    drop = tree.child("drop").rng().random((n, steps)) < dev.p_drop
    spike_hit = tree.child("spike", "hit").rng().random((n, steps)) < dev.spike_prob
    spike_val = tree.child("spike", "val").rng().standard_normal((n, steps, 2)) * dev.spike_std

    outage = seg == TUNNEL_SEG_ID
    lo_k, hi_k = dev.outage_windows
    lo_l, hi_l = dev.outage_len
    rng_win = tree.child("outage").rng()
    for i in range(n):
        k = int(rng_win.integers(lo_k, hi_k + 1))
        for _ in range(k):
            start = int(rng_win.integers(0, steps))
            length = int(rng_win.integers(lo_l, hi_l + 1))
            outage[i, start : start + length] = True

    z = pos_next + np.asarray(dev.gnss_bias) + noise + spike_hit[..., None] * spike_val
    z_mask = ~(drop | outage)
    z = np.where(z_mask[..., None], z, np.nan)
    return z, z_mask, outage, spike_hit
