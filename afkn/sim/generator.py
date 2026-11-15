"""Trajectory generation: route planning, open-loop control, state roll-out, sensors.

Pipeline for one batch (each stage reads its own named child stream, so stages
never perturb each other)::

    plan_routes    -> segment ids, speed targets, yaw-rate commands   (stream "layout")
    plan_controls  -> u_true = [yaw_rate, accel], respecting limits   (deterministic)
    rollout_states -> x with additive process noise Q                 (streams "init", "motion")
    apply_imu / apply_gnss -> measurements                            (streams "imu", "gnss")

The plan is open loop: the speed controller sees the *planned* (noise-free)
speed, not the noisy true speed, so process noise is a genuine unmodelled
disturbance for the filters.
"""

from __future__ import annotations

import numpy as np

from afkn.seeding import SeedTree
from afkn.sim import motion
from afkn.sim.batch import TrajectoryBatch
from afkn.sim.config import SEGMENT_TYPES, SegmentSpec, SimConfig
from afkn.sim.sensors import DeviceParams, apply_gnss, apply_imu

_WANDER_WINDOW = 30  # moving-average length (steps) for the smooth yaw-rate wander


def _fill_speed_targets(out: np.ndarray, spec: SegmentSpec, speed_scale: float, min_speed: float, rng: np.random.Generator) -> None:
    length = out.shape[0]
    lo, hi = spec.speed_block_range
    c = 0
    while c < length:
        block = int(rng.integers(lo, hi + 1))
        out[c : c + block] = max(rng.uniform(*spec.speed_range) * speed_scale, min_speed)
        c += block


def _yaw_profile(length: int, spec: SegmentSpec, turn_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth wander plus raised-cosine turn pulses of random sign."""
    m = _WANDER_WINDOW
    white = rng.standard_normal(length + m - 1)
    w = np.convolve(white, np.ones(m) / m, mode="valid") * np.sqrt(m) * spec.yaw_wander_std
    if spec.turn_rate_range[1] > 0.0 and turn_scale > 0.0:
        len_lo, len_hi = spec.turn_len_range
        gap_lo, gap_hi = spec.turn_gap_range
        c = int(rng.integers(0, gap_hi + 1))
        while True:
            ln = int(rng.integers(len_lo, len_hi + 1))
            if c + ln > length:
                break
            amp = rng.uniform(*spec.turn_rate_range) * turn_scale * (1.0 if rng.random() < 0.5 else -1.0)
            phase = (np.arange(ln) + 0.5) / ln
            w[c : c + ln] += amp * 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
            c += ln + int(rng.integers(gap_lo, gap_hi + 1))
    return w


def plan_routes(
    cfg: SimConfig,
    route_mix: tuple[float, ...],
    speed_scale: float,
    turn_scale: float,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lay out ``n`` routes of ``cfg.steps`` steps. Returns ``(seg, v_target, w_cmd)``, each ``(n, T)``."""
    steps = cfg.steps
    p = np.asarray(route_mix, dtype=float)
    p = p / p.sum()
    seg = np.empty((n, steps), dtype=np.int8)
    v_target = np.empty((n, steps))
    w_cmd = np.zeros((n, steps))
    lo, hi = cfg.seg_len_range
    for i in range(n):
        t = 0
        while t < steps:
            k = int(rng.choice(len(SEGMENT_TYPES), p=p))
            spec = cfg.segments[SEGMENT_TYPES[k]]
            length = min(int(rng.integers(lo, hi + 1)), steps - t)
            seg[i, t : t + length] = k
            _fill_speed_targets(v_target[i, t : t + length], spec, speed_scale, cfg.min_speed, rng)
            w_cmd[i, t : t + length] = _yaw_profile(length, spec, turn_scale, rng)
            t += length
    return seg, v_target, w_cmd


def plan_controls(cfg: SimConfig, v_target: np.ndarray, w_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn targets/commands into feasible controls.

    First-order speed controller with ``|a| <= accel_limit``; yaw rate clipped to
    ``min(v*tan(delta_max)/L, a_lat_max/v)`` (steering and lateral-acceleration limits).
    Returns ``(u_true, v_plan)``: ``(n, T, 2)`` and the planned speed at the start of each step ``(n, T)``.
    """
    n, steps = v_target.shape
    tan_max = np.tan(np.radians(cfg.max_steer_deg))
    v = v_target[:, 0].copy()
    u = np.zeros((n, steps, 2))
    v_plan = np.zeros((n, steps))
    for t in range(steps):
        a = np.clip((v_target[:, t] - v) / cfg.speed_tau, -cfg.accel_limit, cfg.accel_limit)
        w_max = np.minimum(v * tan_max / cfg.wheelbase, cfg.max_lateral_accel / np.maximum(v, cfg.min_speed))
        u[:, t, 0] = np.clip(w_cmd[:, t], -w_max, w_max)
        u[:, t, 1] = a
        v_plan[:, t] = v
        v = v + a * cfg.dt
    return u, v_plan


def rollout_states(cfg: SimConfig, x0: np.ndarray, u_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """``x[t+1] = f(x[t], u[t]) + w[t]``, ``w ~ N(0, Q)``. Returns ``(n, T+1, 4)``."""
    n, steps, _ = u_true.shape
    p = cfg.process_noise
    w = rng.standard_normal((n, steps, 4)) * np.array([p.pos, p.pos, p.heading, p.speed])
    x = np.empty((n, steps + 1, 4))
    x[:, 0] = x0
    for t in range(steps):
        x[:, t + 1] = motion.f(x[:, t], u_true[:, t], cfg.dt, cfg.motion_model) + w[:, t]
    return x


def generate_batch(
    cfg: SimConfig,
    device: DeviceParams,
    route_mix: tuple[float, ...],
    n: int,
    tree: SeedTree,
    *,
    speed_scale: float = 1.0,
    turn_scale: float = 1.0,
    profile_idx: int = 0,
) -> TrajectoryBatch:
    """Generate ``n`` trajectories for one device. Pure function of ``(cfg, device, args, tree)``."""
    if n < 1:
        raise ValueError("n must be >= 1")
    seg, v_target, w_cmd = plan_routes(cfg, route_mix, speed_scale, turn_scale, n, tree.child("layout").rng())
    u_true, _ = plan_controls(cfg, v_target, w_cmd)

    rng_init = tree.child("init").rng()
    x0 = np.zeros((n, 4))
    x0[:, :2] = rng_init.normal(0.0, cfg.start_pos_std, size=(n, 2))
    x0[:, 2] = rng_init.uniform(-np.pi, np.pi, size=n)
    x0[:, 3] = v_target[:, 0]
    init_std = np.array([cfg.init_noise.pos, cfg.init_noise.pos, cfg.init_noise.heading, cfg.init_noise.speed])
    x0_est = x0 + rng_init.standard_normal((n, 4)) * init_std

    x = rollout_states(cfg, x0, u_true, tree.child("motion").rng())
    u_imu, imu_bias = apply_imu(u_true, device, cfg.dt, tree.child("imu"))
    z, z_mask, outage, spike = apply_gnss(x[:, 1:, :2], seg, device, tree.child("gnss"))

    return TrajectoryBatch(
        x=x,
        u_true=u_true,
        u_imu=u_imu,
        z=z,
        z_mask=z_mask,
        seg=seg,
        x0_est=x0_est,
        imu_bias=imu_bias,
        gnss_fault=spike,
        outage=outage,
        profile_idx=np.full(n, profile_idx, dtype=np.int8),
        dt=cfg.dt,
        model=cfg.motion_model,
    )
