"""KalmanNet configuration. Torch-free on purpose: importable and testable without PyTorch.

All values are STARTING DEFAULTS to tune; none has been tuned against any result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from afkn.sim.config import ConfigError, build_dataclass, deep_merge

#: 2 (innovation) + 3 x 4 (state differences) + 1 (GNSS-available flag)
N_FEATURES = 15
#: The gain K is 4 x 2 (state dim x measurement dim), emitted flattened row-major.
N_GAIN = 8


@dataclass(frozen=True)
class KalmanNetSpec:
    """Architecture and input/output scaling of the gain network."""

    hidden: int = 64  # GRU hidden size (project default)
    fc_hidden: int = 64  # width of the output MLP's hidden layer
    innov_scale: float = 3.0  # [m] innovation scale before the asinh squashing
    state_scale: tuple[float, float, float, float] = (3.0, 3.0, 0.05, 0.25)  # [m, m, rad, m/s]
    gain_scale: tuple[float, float, float, float] = (1.0, 1.0, 0.05, 0.5)  # per-row scale of K
    init_pos_gain: float = 0.0  # initial diagonal position gain (0 = start as pure dead reckoning)
    out_init_scale: float = 0.01  # shrink factor for the output layer's initial weights

    def __post_init__(self) -> None:
        if self.hidden < 1 or self.fc_hidden < 1:
            raise ConfigError("hidden sizes must be >= 1")
        if self.innov_scale <= 0 or min(self.state_scale) <= 0 or min(self.gain_scale) <= 0:
            raise ConfigError("scales must be > 0")
        if abs(self.init_pos_gain) >= min(self.gain_scale[0], self.gain_scale[1]):
            raise ConfigError("|init_pos_gain| must be < the position gain scales (the gain is tanh-bounded)")
        if self.out_init_scale < 0:
            raise ConfigError("out_init_scale must be >= 0")


@dataclass(frozen=True)
class TrainSpec:
    """Training hyper-parameters (project defaults: Adam, lr 1e-3, gradient clip 1.0)."""

    lr: float = 1e-3
    grad_clip: float = 1.0
    batch_size: int = 16
    chunk_len: int = 50  # truncated-BPTT window (steps); one optimiser step per chunk
    epochs: int = 40
    weight_decay: float = 0.0
    loss_weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)  # MSE weights on [px, py, heading, speed]

    def __post_init__(self) -> None:
        if self.lr <= 0 or self.grad_clip <= 0:
            raise ConfigError("lr and grad_clip must be > 0")
        if self.batch_size < 1 or self.chunk_len < 1 or self.epochs < 0:
            raise ConfigError("batch_size and chunk_len must be >= 1, epochs >= 0")
        if min(self.loss_weights) < 0 or sum(self.loss_weights) <= 0:
            raise ConfigError("loss_weights must be >= 0 with positive sum")


@dataclass(frozen=True)
class KalmanNetConfig:
    net: KalmanNetSpec = field(default_factory=KalmanNetSpec)
    train: TrainSpec = field(default_factory=TrainSpec)


def expected_num_params(spec: KalmanNetSpec) -> int:
    """Parameter count implied by the architecture, computed without PyTorch.

    fc_in (N_FEATURES -> hidden) + GRUCell(hidden -> hidden) + fc1 (hidden -> fc_hidden)
    + fc2 (fc_hidden -> N_GAIN). The default spec gives 30,664.
    """
    h, fc = spec.hidden, spec.fc_hidden
    fc_in = N_FEATURES * h + h
    gru = 3 * h * h + 3 * h * h + 3 * h + 3 * h  # weight_ih, weight_hh, bias_ih, bias_hh
    fc1 = h * fc + fc
    fc2 = fc * N_GAIN + N_GAIN
    return fc_in + gru + fc1 + fc2


def load_kalmannet_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> KalmanNetConfig:
    """Load from YAML (default: packaged ``kalmannet.yaml``) with optional deep-merged overrides."""
    if path is None:
        text = resources.files("afkn.configs").joinpath("kalmannet.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if overrides:
        raw = deep_merge(raw, overrides)
    return build_dataclass(KalmanNetConfig, raw, "kalmannet")
