"""Sensor statistics: empirical noise must match the configured sigma (and the other error terms).

Tolerances are chosen from sample-size standard errors, not tuned to pass:
for n i.i.d. normal samples the relative s.e. of the sample std is 1/sqrt(2n);
with n >= 20k samples a 3 % tolerance is > 6 s.e.
"""

from __future__ import annotations

import numpy as np
import pytest

from afkn.seeding import SeedTree
from afkn.sim.config import SimConfig, load_config
from afkn.sim.generator import generate_batch
from afkn.sim.sensors import realize_device

N, STEPS = 60, 500
MIX = (0.5, 0.35, 0.15)
PROFILES = ("high_grade", "mid_grade", "low_grade", "intermittent", "faulty")


def make(cfg: SimConfig, profile: str, *, seed: int = 0, n: int = N, steps: int = STEPS, mix=MIX, fault_mode=None, sigma_scale=1.0):
    cfg = cfg.replace(steps=steps)
    dev = realize_device(profile, cfg.profiles[profile], SeedTree(seed).child("dev").rng(), sigma_scale=sigma_scale, fault_mode=fault_mode)
    return dev, generate_batch(cfg, dev, mix, n, SeedTree(seed).child("batch"))


def assert_std(sample: np.ndarray, sigma: float, rel: float = 0.03) -> None:
    assert sample.size > 10_000
    assert abs(sample.std() / sigma - 1.0) < rel, f"std {sample.std():.5f} vs sigma {sigma:.5f}"


def assert_zero_mean(sample: np.ndarray, sigma: float) -> None:
    assert abs(sample.mean()) < 5.0 * sigma / np.sqrt(sample.size)


# --------------------------------------------------------------------------- GNSS


@pytest.mark.parametrize(
    "profile, fault_mode",
    [(p, None) for p in PROFILES if p != "faulty"] + [("faulty", "constant_bias"), ("faulty", "random_spikes")],
)
def test_gnss_noise_std_matches_configured_sigma(default_cfg, profile, fault_mode):
    dev, b = make(default_cfg, profile, fault_mode=fault_mode)
    ok = b.z_mask & ~b.gnss_fault
    resid = b.z[ok] - b.x[:, 1:, :2][ok] - np.asarray(dev.gnss_bias)
    for axis in (0, 1):
        assert_std(resid[:, axis], dev.gnss_sigma)
        assert_zero_mean(resid[:, axis], dev.gnss_sigma)


def test_gnss_constant_bias_is_recovered(default_cfg):
    # Pick a profile whose bias is large relative to its noise so the test can discriminate.
    cfg = load_config(overrides={"profiles": {"low_grade": {"gnss": {"bias_std": 3.0}}}})
    dev, b = make(cfg, "low_grade")
    assert max(abs(v) for v in dev.gnss_bias) > 1.0
    ok = b.z_mask
    raw = b.z[ok] - b.x[:, 1:, :2][ok]
    assert np.allclose(raw.mean(axis=0), dev.gnss_bias, atol=5.0 * dev.gnss_sigma / np.sqrt(len(raw)))


def test_z_measures_the_NEXT_state(default_cfg):
    """Time convention: z[t] observes x[t+1] (not x[t])."""
    _, b = make(default_cfg, "high_grade", n=20)
    ok = b.z_mask
    err_next = np.linalg.norm(b.z[ok] - b.x[:, 1:, :2][ok], axis=1).mean()
    err_prev = np.linalg.norm(b.z[ok] - b.x[:, :-1, :2][ok], axis=1).mean()
    assert err_next < 1.5  # ~ sigma * sqrt(pi/2) for sigma = 0.5 plus a small bias
    assert err_prev > 3.0 * err_next


# ---------------------------------------------------------------------- availability


def test_dropout_rate_matches_p_drop(default_cfg):
    dev, b = make(default_cfg, "low_grade")
    free = ~b.outage
    rate = 1.0 - b.z_mask[free].mean()
    se = np.sqrt(dev.p_drop * (1 - dev.p_drop) / free.sum())
    assert abs(rate - dev.p_drop) < 5.0 * se


def test_unavailable_samples_are_nan_and_only_those(default_cfg):
    _, b = make(default_cfg, "intermittent", n=20)
    assert np.array_equal(np.isnan(b.z).any(axis=-1), ~b.z_mask)
    assert np.array_equal(np.isnan(b.z).all(axis=-1), ~b.z_mask)


def test_tunnel_segments_force_gnss_outage(default_cfg):
    _, b = make(default_cfg, "high_grade", mix=(0.0, 0.0, 1.0), n=10)
    assert (b.seg == 2).all() and b.outage.all()
    assert not b.z_mask.any() and np.isnan(b.z).all()
    _, b = make(default_cfg, "high_grade", mix=(0.3, 0.3, 0.4), n=40)
    assert not b.z_mask[b.seg == 2].any()  # every tunnel step is an outage
    assert b.z_mask[b.seg != 2].mean() > 0.9  # and high-grade is otherwise mostly available


def test_non_intermittent_profiles_have_no_random_outage_windows(default_cfg):
    _, b = make(default_cfg, "high_grade", mix=(1.0, 0.0, 0.0), n=20)
    assert not b.outage.any()


def test_intermittent_profile_has_contiguous_outage_windows(default_cfg):
    dev, b = make(default_cfg, "intermittent", mix=(1.0, 0.0, 0.0), n=40)
    assert 0.15 < b.outage.mean() < 0.85
    lo, _ = dev.outage_len
    for row in b.outage:
        edges = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            if e < len(row):  # runs cut off by the end of the trajectory may be short
                assert e - s >= lo


def test_outage_windows_count_within_configured_range(default_cfg):
    cfg = load_config(overrides={"profiles": {"intermittent": {"gnss": {"outage_windows": [2, 2], "outage_len": [10, 10]}}}})
    _, b = make(cfg, "intermittent", mix=(1.0, 0.0, 0.0), n=30, steps=1000)
    per_traj = b.outage.sum(axis=1)
    assert per_traj.max() <= 20 and per_traj.min() >= 10  # two windows of 10 (may overlap / touch)


# ------------------------------------------------------------------------- faults


def test_faulty_constant_bias_mode(default_cfg):
    dev, b = make(default_cfg, "faulty", fault_mode="constant_bias")
    assert dev.fault_mode == "constant_bias" and dev.spike_prob == 0.0
    assert abs(np.hypot(*dev.gnss_bias) - 15.0) < 2.0  # fault magnitude +/- calibration bias
    assert abs(abs(dev.imu_bias0[0]) - 0.05) < 0.03
    assert not b.gnss_fault.any()
    ok = b.z_mask
    mean_err = (b.z[ok] - b.x[:, 1:, :2][ok]).mean(axis=0)
    assert np.allclose(mean_err, dev.gnss_bias, atol=0.1)


def test_faulty_random_spikes_mode(default_cfg):
    dev, b = make(default_cfg, "faulty", fault_mode="random_spikes")
    assert dev.fault_mode == "random_spikes" and np.hypot(*dev.gnss_bias) < 2.0  # no constant offset
    p = dev.spike_prob
    assert abs(b.gnss_fault.mean() - p) < 5.0 * np.sqrt(p * (1 - p) / b.gnss_fault.size)
    resid = b.z - b.x[:, 1:, :2] - np.asarray(dev.gnss_bias)
    hit = b.z_mask & b.gnss_fault
    assert hit.sum() > 500
    assert abs(resid[hit].std() / np.hypot(dev.gnss_sigma, dev.spike_std) - 1.0) < 0.1
    assert abs(resid[b.z_mask & ~b.gnss_fault].std() / dev.gnss_sigma - 1.0) < 0.03


def test_honest_profiles_have_no_fault(default_cfg):
    for p in ("high_grade", "mid_grade", "low_grade", "intermittent"):
        dev, b = make(default_cfg, p, n=5, steps=100)
        assert dev.fault_mode is None and dev.spike_prob == 0.0 and not b.gnss_fault.any()


# ---------------------------------------------------------------------------- IMU


@pytest.mark.parametrize("profile", ["high_grade", "mid_grade", "low_grade"])
def test_imu_white_noise_std_matches_configured_sigma(default_cfg, profile):
    dev, b = make(default_cfg, profile)
    resid = b.u_imu - b.u_true * (1.0 + np.asarray(dev.imu_scale)) - b.imu_bias
    for axis in (0, 1):
        assert_std(resid[..., axis].ravel(), dev.imu_sigma[axis])
        assert_zero_mean(resid[..., axis].ravel(), dev.imu_sigma[axis])


@pytest.mark.parametrize("profile", ["mid_grade", "low_grade"])
def test_imu_bias_random_walk_increment_std(default_cfg, profile):
    dev, b = make(default_cfg, profile)
    dt = default_cfg.dt
    inc = np.diff(b.imu_bias, axis=1)
    for axis in (0, 1):
        assert_std(inc[..., axis].ravel(), dev.imu_rw[axis] * np.sqrt(dt), rel=0.03)
    # The walk accumulates: Var[bias_T - bias_0] ~ rw^2 * dt * (T-1)
    drift = b.imu_bias[:, -1, :] - b.imu_bias[:, 0, :]
    expected = np.asarray(dev.imu_rw) * np.sqrt(dt * (STEPS - 1))
    assert np.all(np.abs(drift.std(axis=0) / expected - 1.0) < 0.35)  # n=60 trajectories -> ~9 % s.e.


def test_imu_bias_starts_at_device_constant(default_cfg):
    dev, b = make(default_cfg, "low_grade", n=200, steps=50)
    first = b.imu_bias[:, 0, :]
    se = np.asarray(dev.imu_rw) * np.sqrt(default_cfg.dt) / np.sqrt(len(first))
    assert np.all(np.abs(first.mean(axis=0) - np.asarray(dev.imu_bias0)) < 6.0 * se)


def test_imu_scale_error_is_applied(default_cfg):
    cfg = load_config(overrides={"profiles": {"low_grade": {"imu": {"scale_std": 0.2}}}})
    dev, b = make(cfg, "low_grade")
    assert max(abs(s) for s in dev.imu_scale) > 0.05  # big enough to be detectable
    for axis in (0, 1):
        u = b.u_true[..., axis].ravel()
        y = (b.u_imu[..., axis] - b.imu_bias[..., axis]).ravel()
        slope = (u @ y) / (u @ u)
        se = dev.imu_sigma[axis] / np.sqrt(u @ u)
        assert abs(slope - (1.0 + dev.imu_scale[axis])) < 5.0 * se
        assert abs(slope - 1.0) > 5.0 * se  # ...and clearly different from "no scale error"


# ------------------------------------------------------------------- device / seeds


def test_sigma_scale_scales_noise_but_not_constants(default_cfg):
    prof = default_cfg.profiles["mid_grade"]
    d1 = realize_device("mid_grade", prof, SeedTree(3).rng(), sigma_scale=1.0)
    d2 = realize_device("mid_grade", prof, SeedTree(3).rng(), sigma_scale=2.0)
    assert np.isclose(d2.gnss_sigma, 2 * d1.gnss_sigma)
    assert np.allclose(d2.imu_sigma, 2 * np.asarray(d1.imu_sigma))
    assert np.allclose(d2.imu_rw, 2 * np.asarray(d1.imu_rw))
    assert d1.gnss_bias == d2.gnss_bias and d1.imu_bias0 == d2.imu_bias0 and d1.imu_scale == d2.imu_scale


def test_R_matches_effective_sigma(default_cfg):
    dev = realize_device("low_grade", default_cfg.profiles["low_grade"], SeedTree(0).rng(), sigma_scale=1.15)
    assert np.allclose(dev.R(), (4.0 * 1.15) ** 2 * np.eye(2))


def test_changing_p_drop_does_not_shift_other_noise(default_cfg):
    """Fixed draw counts: dropout settings must not change the noise on samples that remain."""
    cfg2 = load_config(overrides={"profiles": {"mid_grade": {"gnss": {"p_drop": 0.3}}}})
    _, a = make(default_cfg, "mid_grade", n=10, steps=200)
    _, b = make(cfg2, "mid_grade", n=10, steps=200)
    both = a.z_mask & b.z_mask
    assert both.sum() > 0 and not np.array_equal(a.z_mask, b.z_mask)
    assert np.array_equal(a.z[both], b.z[both])
    assert np.array_equal(a.x, b.x) and np.array_equal(a.u_imu, b.u_imu)
