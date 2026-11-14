"""Frozen, validated configuration dataclasses and the YAML loader.

All simulation behaviour is config-driven. Ablations are config flags, not code
forks. Every default in ``afkn/configs/default.yaml`` is a *starting default to
tune*; none of them has been tuned against results.

Conventions
-----------
* ``route_mix`` order is ``SEGMENT_TYPES = (urban, highway, tunnel)``.
* Sequences are tuples (immutable); the three mapping fields on ``SimConfig``
  (``segments``, ``profiles``) are plain dicts that must be treated as read-only.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import types
import typing
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np
import yaml

SEGMENT_TYPES: tuple[str, ...] = ("urban", "highway", "tunnel")
MOTION_MODELS: tuple[str, ...] = ("ctrv", "bicycle")
FAULT_MODES: tuple[str, ...] = ("constant_bias", "random_spikes")


class ConfigError(ValueError):
    """Raised for any invalid or unknown configuration content."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def _range_ok(r: tuple[float, float], name: str, *, integer: bool = False, minimum: float = 0.0) -> None:
    _require(len(r) == 2, f"{name} must have two entries")
    _require(r[0] <= r[1], f"{name} must satisfy lo <= hi, got {r}")
    _require(r[0] >= minimum, f"{name} must be >= {minimum}, got {r}")
    if integer:
        _require(all(float(v).is_integer() for v in r), f"{name} must be integers, got {r}")


# --------------------------------------------------------------------------- noise


@dataclass(frozen=True)
class ProcessNoiseSpec:
    """Per-step additive process-noise std on [px, py, heading, speed]; Q = diag(std^2)."""

    pos: float = 0.02
    heading: float = 0.002
    speed: float = 0.01

    def __post_init__(self) -> None:
        _require(min(self.pos, self.heading, self.speed) >= 0, "process noise std must be >= 0")


@dataclass(frozen=True)
class InitNoiseSpec:
    """Std of the initial-state *estimate* error; P0 = diag(std^2)."""

    pos: float = 1.0
    heading: float = 0.05
    speed: float = 0.5

    def __post_init__(self) -> None:
        _require(min(self.pos, self.heading, self.speed) >= 0, "init noise std must be >= 0")


# ------------------------------------------------------------------------ segments


@dataclass(frozen=True)
class SegmentSpec:
    """Behaviour of one route-segment type (urban / highway / tunnel)."""

    speed_range: tuple[float, float]
    speed_block_range: tuple[int, int] = (50, 150)
    turn_rate_range: tuple[float, float] = (0.0, 0.0)
    turn_len_range: tuple[int, int] = (1, 1)
    turn_gap_range: tuple[int, int] = (1, 1)
    yaw_wander_std: float = 0.0

    def __post_init__(self) -> None:
        _range_ok(self.speed_range, "speed_range", minimum=0.0)
        _range_ok(self.speed_block_range, "speed_block_range", integer=True, minimum=1)
        _range_ok(self.turn_rate_range, "turn_rate_range", minimum=0.0)
        _range_ok(self.turn_len_range, "turn_len_range", integer=True, minimum=1)
        _range_ok(self.turn_gap_range, "turn_gap_range", integer=True, minimum=1)
        _require(self.yaw_wander_std >= 0, "yaw_wander_std must be >= 0")


# ------------------------------------------------------------------------- sensors


@dataclass(frozen=True)
class GNSSSpec:
    """GNSS: z = [px, py] + device_bias + N(0, sigma^2 I), with dropout and outages."""

    sigma: float = 1.0
    bias_std: float = 0.0  # std of the per-device constant calibration bias (per axis)
    p_drop: float = 0.0  # i.i.d. per-sample dropout probability
    outage_windows: tuple[int, int] = (0, 0)  # number of random outage windows per trajectory
    outage_len: tuple[int, int] = (1, 1)  # window length in steps

    def __post_init__(self) -> None:
        _require(self.sigma > 0, "gnss.sigma must be > 0")
        _require(self.bias_std >= 0, "gnss.bias_std must be >= 0")
        _require(0.0 <= self.p_drop <= 1.0, "gnss.p_drop must be in [0, 1]")
        _range_ok(self.outage_windows, "gnss.outage_windows", integer=True)
        _range_ok(self.outage_len, "gnss.outage_len", integer=True, minimum=1)


@dataclass(frozen=True)
class IMUSpec:
    """IMU yaw-rate [rad/s] and acceleration [m/s^2] measurement errors.

    u_imu = (1 + scale) * u_true + bias(t) + N(0, sigma^2), with
    bias(t) = bias0 + random-walk(rw * sqrt(dt)). ``bias0`` and ``scale`` are
    per-device constants; the random walk restarts at 0 on every trajectory.
    """

    sigma_yaw: float = 0.01
    sigma_acc: float = 0.1
    bias0_std_yaw: float = 0.0
    bias0_std_acc: float = 0.0
    rw_yaw: float = 0.0  # rad/s per sqrt(s)
    rw_acc: float = 0.0  # m/s^2 per sqrt(s)
    scale_std: float = 0.0  # std of the per-device multiplicative scale error (per axis)

    def __post_init__(self) -> None:
        _require(self.sigma_yaw > 0 and self.sigma_acc > 0, "imu sigmas must be > 0")
        vals = (self.bias0_std_yaw, self.bias0_std_acc, self.rw_yaw, self.rw_acc, self.scale_std)
        _require(min(vals) >= 0, "imu bias/rw/scale stds must be >= 0")


@dataclass(frozen=True)
class FaultSpec:
    """Fault parameters for a faulty device. ``mode`` selects which block applies.

    constant_bias: GNSS offset of ``gnss_bias_m`` metres in a random (per-device)
        direction plus a ``imu_yaw_bias`` offset with random sign.
    random_spikes: each GNSS sample gets, with probability ``spike_prob``, an
        extra N(0, spike_std^2 I) outlier.
    """

    mode: str | None = None
    gnss_bias_m: float = 0.0
    imu_yaw_bias: float = 0.0
    spike_prob: float = 0.0
    spike_std: float = 0.0

    def __post_init__(self) -> None:
        _require(self.mode is None or self.mode in FAULT_MODES, f"fault.mode must be None or in {FAULT_MODES}")
        _require(self.gnss_bias_m >= 0 and self.imu_yaw_bias >= 0, "fault biases must be >= 0")
        _require(0.0 <= self.spike_prob <= 1.0 and self.spike_std >= 0, "invalid spike parameters")


@dataclass(frozen=True)
class ProfileSpec:
    """One of the five client sensor profiles."""

    gnss: GNSSSpec = field(default_factory=GNSSSpec)
    imu: IMUSpec = field(default_factory=IMUSpec)
    fault: FaultSpec = field(default_factory=FaultSpec)


@dataclass(frozen=True)
class ClientSpec:
    """Per-client heterogeneity (non-IID): profile, routes, speeds, noise level."""

    profile: str
    route_mix: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # (urban, highway, tunnel)
    speed_scale: float = 1.0
    turn_scale: float = 1.0
    sigma_scale: float = 1.0  # multiplies GNSS sigma, IMU white-noise sigmas and IMU random-walk rates
    fault_mode: str | None = None  # overrides the profile's fault mode when set

    def __post_init__(self) -> None:
        _require(len(self.route_mix) == len(SEGMENT_TYPES), f"route_mix needs {len(SEGMENT_TYPES)} entries")
        _require(min(self.route_mix) >= 0 and sum(self.route_mix) > 0, "route_mix must be >= 0 with positive sum")
        _require(self.speed_scale > 0 and self.turn_scale >= 0 and self.sigma_scale > 0, "invalid client scales")
        _require(self.fault_mode is None or self.fault_mode in FAULT_MODES, f"fault_mode must be None or in {FAULT_MODES}")


@dataclass(frozen=True)
class SplitSpec:
    """Trajectories per client split, plus the shared global-test and server-trusted sets."""

    train: int = 200
    val: int = 40
    test: int = 40
    global_test: int = 200
    trusted: int = 200

    def __post_init__(self) -> None:
        _require(min(self.train, self.val, self.test, self.global_test, self.trusted) >= 0, "split sizes must be >= 0")


# ---------------------------------------------------------------------- top level


@dataclass(frozen=True)
class SimConfig:
    """Complete simulation + dataset configuration."""

    seed: int = 0
    dt: float = 0.1
    steps: int = 1000
    motion_model: str = "ctrv"  # "ctrv" (exact arc) or "bicycle" (midpoint-heading Euler)
    wheelbase: float = 2.7  # [m] used for the steering-angle limit
    max_steer_deg: float = 35.0
    max_lateral_accel: float = 4.0  # [m/s^2]
    accel_limit: float = 2.5  # [m/s^2] longitudinal command limit
    speed_tau: float = 3.0  # [s] speed-controller time constant
    min_speed: float = 1.0  # [m/s] floor for speed targets
    start_pos_std: float = 50.0
    process_noise: ProcessNoiseSpec = field(default_factory=ProcessNoiseSpec)
    init_noise: InitNoiseSpec = field(default_factory=InitNoiseSpec)
    seg_len_range: tuple[int, int] = (100, 300)
    segments: dict[str, SegmentSpec] = field(default_factory=dict)
    profiles: dict[str, ProfileSpec] = field(default_factory=dict)
    clients: tuple[ClientSpec, ...] = ()
    splits: SplitSpec = field(default_factory=SplitSpec)
    mixed_route_mix: tuple[float, float, float] = (0.5, 0.35, 0.15)  # global-test / trusted sets

    def __post_init__(self) -> None:
        _require(self.dt > 0 and self.steps >= 1, "dt must be > 0 and steps >= 1")
        _require(self.seed >= 0, "seed must be >= 0")
        _require(self.motion_model in MOTION_MODELS, f"motion_model must be one of {MOTION_MODELS}")
        _require(self.wheelbase > 0 and 0 < self.max_steer_deg < 90, "invalid wheelbase / max_steer_deg")
        _require(min(self.max_lateral_accel, self.accel_limit, self.speed_tau, self.min_speed) > 0, "limits must be > 0")
        _require(self.start_pos_std >= 0, "start_pos_std must be >= 0")
        _range_ok(self.seg_len_range, "seg_len_range", integer=True, minimum=1)
        _require(set(self.segments) == set(SEGMENT_TYPES), f"segments must define exactly {SEGMENT_TYPES}")
        _require(len(self.profiles) >= 1, "at least one profile is required")
        _require(len(self.clients) >= 1, "at least one client is required")
        for i, c in enumerate(self.clients):
            _require(c.profile in self.profiles, f"clients[{i}].profile '{c.profile}' is not a defined profile")
        _require(
            len(self.mixed_route_mix) == len(SEGMENT_TYPES)
            and min(self.mixed_route_mix) >= 0
            and sum(self.mixed_route_mix) > 0,
            "mixed_route_mix must have 3 non-negative entries with positive sum",
        )

    # -- derived quantities used by filters / tests --------------------------------
    @property
    def profile_names(self) -> tuple[str, ...]:
        """Profile names in definition order; ``profile_idx`` in batches indexes this."""
        return tuple(self.profiles)

    def Q(self) -> np.ndarray:
        """Process-noise covariance matching the simulator (4x4)."""
        p = self.process_noise
        return np.diag([p.pos**2, p.pos**2, p.heading**2, p.speed**2])

    def P0(self) -> np.ndarray:
        """Initial-estimate covariance matching the simulator (4x4)."""
        p = self.init_noise
        return np.diag([p.pos**2, p.pos**2, p.heading**2, p.speed**2])

    def replace(self, **changes: Any) -> "SimConfig":
        """Return a validated copy with fields replaced."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON form (logged with every experiment)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=list)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- YAML loader


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp):
        _require(isinstance(value, dict), f"{where}: expected a mapping")
        return _build(tp, value, where)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        args = [a for a in get_args(tp) if a is not type(None)]
        return _coerce(args[0], value, where)
    if origin is tuple:
        args = get_args(tp)
        _require(isinstance(value, (list, tuple)), f"{where}: expected a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{where}[{i}]") for i, v in enumerate(value))
        _require(len(value) == len(args), f"{where}: expected {len(args)} entries, got {len(value)}")
        return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
    if origin is dict:
        _require(isinstance(value, dict), f"{where}: expected a mapping")
        _, vt = get_args(tp)
        return {str(k): _coerce(vt, v, f"{where}.{k}") for k, v in value.items()}
    if tp is bool:
        _require(isinstance(value, bool), f"{where}: expected a bool")
        return value
    if tp is int:
        _require(not isinstance(value, bool) and isinstance(value, (int, float)) and float(value).is_integer(),
                 f"{where}: expected an integer, got {value!r}")
        return int(value)
    if tp is float:
        _require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        _require(isinstance(value, str), f"{where}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: unsupported field type {tp!r}")


def _build(cls: type, raw: dict[str, Any], where: str) -> Any:
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = set(raw) - names
    _require(not unknown, f"{where}: unknown key(s) {sorted(unknown)}; valid keys are {sorted(names)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in raw.items()}
    try:
        return cls(**kwargs)
    except TypeError as exc:  # missing required field
        raise ConfigError(f"{where}: {exc}") from exc


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base`` (lists/scalars are replaced)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> SimConfig:
    """Load a ``SimConfig`` from YAML (default: the packaged ``default.yaml``).

    ``overrides`` is deep-merged over the YAML before validation, which is how
    ablation flags are applied from the command line or from tests.
    """
    if path is None:
        text = resources.files("afkn.configs").joinpath("default.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    _require(isinstance(raw, dict), "config root must be a mapping")
    if overrides:
        raw = deep_merge(raw, overrides)
    return _build(SimConfig, raw, "config")
