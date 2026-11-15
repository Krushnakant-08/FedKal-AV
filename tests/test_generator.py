"""Trajectory generator: process/initial noise, planning limits, route mix, determinism, data contract."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.seeding import SeedTree
from afkn.sim import motion
from afkn.sim.batch import FIELD_NAMES, TrajectoryBatch
from afkn.sim.config import SEGMENT_TYPES, SimConfig
from afkn.sim.generator import generate_batch, plan_controls, plan_routes
from afkn.sim.sensors import realize_device


def make(cfg: SimConfig, n: int = 40, *, profile="mid_grade", mix=(0.5, 0.35, 0.15), seed=0, **kw) -> TrajectoryBatch:
    dev = realize_device(profile, cfg.profiles[profile], SeedTree(seed).child("dev").rng())
    return generate_batch(cfg, dev, mix, n, SeedTree(seed).child("batch"), **kw)


# ------------------------------------------------------------------- process noise


@pytest.mark.parametrize("model", ["ctrv", "bicycle"])
def test_process_noise_std_matches_configured_Q(default_cfg, model):
    cfg = default_cfg.replace(steps=500, motion_model=model)
    b = make(cfg, n=60)
    w = b.x[:, 1:] - motion.f(b.x[:, :-1], b.u_true, cfg.dt, model)  # exactly the injected noise
    p = cfg.process_noise
    for j, sigma in enumerate([p.pos, p.pos, p.heading, p.speed]):
        s = w[..., j].ravel()
        assert abs(s.std() / sigma - 1.0) < 0.03, f"component {j}"
        assert abs(s.mean()) < 5.0 * sigma / np.sqrt(s.size)
    # Q is diagonal: components are uncorrelated
    corr = np.corrcoef(w.reshape(-1, 4).T)
    assert np.abs(corr - np.eye(4)).max() < 0.03


def test_zero_process_noise_gives_deterministic_dynamics(default_cfg):
    cfg = default_cfg.replace(steps=100, process_noise=default_cfg.process_noise.__class__(pos=0.0, heading=0.0, speed=0.0))
    b = make(cfg, n=5)
    assert np.allclose(b.x[:, 1:], motion.f(b.x[:, :-1], b.u_true, cfg.dt, cfg.motion_model), atol=1e-9)


def test_initial_estimate_error_matches_P0(default_cfg):
    cfg = default_cfg.replace(steps=3)
    b = make(cfg, n=4000)
    err = b.x0_est - b.x[:, 0]
    sig = np.sqrt(np.diag(cfg.P0()))
    assert np.all(np.abs(err.std(axis=0) / sig - 1.0) < 0.05)
    assert np.all(np.abs(err.mean(axis=0)) < 5.0 * sig / np.sqrt(len(err)))


# ----------------------------------------------------------------------- planning


def test_planned_controls_respect_all_limits(default_cfg):
    cfg = default_cfg
    rng = SeedTree(1).rng()
    for mix in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        _, v_target, w_cmd = plan_routes(cfg, mix, 1.0, 2.0, 30, rng)  # turn_scale 2 stresses the limits
        u, v_plan = plan_controls(cfg, v_target, w_cmd)
        w, a = u[..., 0], u[..., 1]
        assert np.abs(a).max() <= cfg.accel_limit + 1e-12
        assert (np.abs(w) * v_plan).max() <= cfg.max_lateral_accel + 1e-9  # lateral acceleration
        steer = motion.steering_angle(w, v_plan, cfg.wheelbase)
        assert np.abs(steer).max() <= np.radians(cfg.max_steer_deg) + 1e-9
        assert v_plan.min() >= cfg.min_speed - 1e-9


def test_limits_actually_bind_when_commands_are_aggressive(default_cfg):
    """Guard against vacuous limit tests: with huge commands the clip must be active."""
    n, T = 5, 200
    v_target = np.full((n, T), 10.0)
    w_cmd = np.full((n, T), 3.0)
    u, v_plan = plan_controls(default_cfg, v_target, w_cmd)
    assert np.allclose(u[..., 0], default_cfg.max_lateral_accel / 10.0)  # 0.4 rad/s, below the steering limit at 10 m/s


def test_speed_targets_follow_segment_ranges_and_client_scale(default_cfg):
    rng = SeedTree(2).rng()
    seg, v_target, _ = plan_routes(default_cfg, (0, 1, 0), 0.9, 1.0, 50, rng)
    lo, hi = default_cfg.segments["highway"].speed_range
    assert (seg == 1).all()
    assert v_target.min() >= lo * 0.9 - 1e-9 and v_target.max() <= hi * 0.9 + 1e-9


def test_route_mix_is_realised(default_cfg):
    rng = SeedTree(3).rng()
    mix = (0.2, 0.7, 0.1)
    seg, _, _ = plan_routes(default_cfg, mix, 1.0, 1.0, 200, rng)
    frac = np.array([(seg == k).mean() for k in range(3)])
    assert np.abs(frac - np.array(mix)).max() < 0.06
    assert set(np.unique(seg)) <= set(range(len(SEGMENT_TYPES)))


def test_urban_turns_highway_gentle_tunnel_straight(default_cfg):
    rng = SeedTree(4).rng()
    peaks = {}
    for name, mix in zip(SEGMENT_TYPES, [(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        _, v_target, w_cmd = plan_routes(default_cfg, mix, 1.0, 1.0, 40, rng)
        u, _ = plan_controls(default_cfg, v_target, w_cmd)
        peaks[name] = np.abs(u[..., 0]).max()
    assert peaks["urban"] > 0.25 > peaks["highway"] > peaks["tunnel"]
    assert peaks["tunnel"] < 0.05


def test_turn_scale_zero_removes_turn_events(default_cfg):
    rng = SeedTree(5).rng()
    _, v_target, w_cmd = plan_routes(default_cfg, (1, 0, 0), 1.0, 0.0, 30, rng)
    u, _ = plan_controls(default_cfg, v_target, w_cmd)
    assert np.abs(u[..., 0]).max() < 0.1  # only the small wander remains


# --------------------------------------------------------------------- determinism


def test_same_tree_bit_identical_different_tree_differs(default_cfg):
    cfg = default_cfg.replace(steps=200)
    a, b, c = make(cfg, 8, seed=7), make(cfg, 8, seed=7), make(cfg, 8, seed=8)
    assert a.content_hash() == b.content_hash()
    assert all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True) for f in FIELD_NAMES if getattr(a, f).dtype.kind == "f")
    assert a.content_hash() != c.content_hash()


def test_motion_model_changes_states_not_plan_or_measurement_noise(default_cfg):
    a = make(default_cfg.replace(steps=300, motion_model="ctrv"), 10)
    b = make(default_cfg.replace(steps=300, motion_model="bicycle"), 10)
    assert np.array_equal(a.u_true, b.u_true) and np.array_equal(a.u_imu, b.u_imu)
    assert np.abs(a.x - b.x).max() > 1e-3  # genuinely different dynamics
    assert a.model == "ctrv" and b.model == "bicycle"


def test_values_are_finite_where_required(default_cfg):
    b = make(default_cfg, 20)
    for name in ("x", "u_true", "u_imu", "x0_est", "imu_bias"):
        assert np.isfinite(getattr(b, name)).all(), name
    assert b.x[..., 3].min() > 0.0  # speed stays positive under the default process noise


# ---------------------------------------------------------------- data contract


def test_batch_shapes_dtypes_and_time_convention(default_cfg):
    b = make(default_cfg, 3)
    assert b.x.shape == (3, 1001, 4) and b.u_true.shape == (3, 1000, 2) and b.z.shape == (3, 1000, 2)
    assert b.z_mask.shape == (3, 1000) and b.dt == 0.1 and b.T == 1000 and b.n == 3
    assert b.x.dtype == np.float64 and b.z_mask.dtype == bool and b.seg.dtype == np.int8


def test_subset_and_concat_roundtrip(default_cfg):
    cfg = default_cfg.replace(steps=60)
    a, b = make(cfg, 6, seed=1), make(cfg, 4, seed=2)
    cat = TrajectoryBatch.concat([a, b])
    assert cat.n == 10
    assert cat.subset(slice(0, 6)).content_hash() == a.content_hash()
    assert cat.subset(slice(6, 10)).content_hash() == b.content_hash()
    perm = np.random.default_rng(0).permutation(10)
    assert np.array_equal(cat.subset(perm).x, cat.x[perm])


def test_content_hash_detects_single_value_change(default_cfg):
    b = make(default_cfg.replace(steps=40), 3)
    h = b.content_hash()
    b.u_imu[1, 5, 0] += 1e-12
    assert b.content_hash() != h


def test_validation_rejects_inconsistent_batches(default_cfg):
    b = make(default_cfg.replace(steps=40), 3)
    kw = {f: getattr(b, f).copy() for f in FIELD_NAMES}
    bad_mask = dict(kw, z_mask=~kw["z_mask"])
    with pytest.raises(ValueError, match="NaN"):
        TrajectoryBatch(**bad_mask, dt=b.dt, model=b.model)
    with pytest.raises(ValueError, match="shape"):
        TrajectoryBatch(**dict(kw, x=kw["x"][:, :-1]), dt=b.dt, model=b.model)
    with pytest.raises(ValueError, match="dtype"):
        TrajectoryBatch(**dict(kw, x=kw["x"].astype(np.float32)), dt=b.dt, model=b.model)
    with pytest.raises(ValueError):
        TrajectoryBatch.concat([])
    with pytest.raises(ValueError, match="share"):
        TrajectoryBatch.concat([b, make(default_cfg.replace(steps=41), 2)])
