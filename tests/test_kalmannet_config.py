"""Torch-free tests for the KalmanNet configuration (run everywhere, including without PyTorch)."""

from __future__ import annotations

import pytest

from afkn.models import (
    N_FEATURES,
    N_GAIN,
    KalmanNetSpec,
    TrainSpec,
    expected_num_params,
    load_kalmannet_config,
)
from afkn.sim.config import ConfigError


def test_defaults_match_project_spec() -> None:
    cfg = load_kalmannet_config()
    assert cfg.net.hidden == 64
    assert cfg.train.lr == 1e-3
    assert cfg.train.grad_clip == 1.0
    assert (N_FEATURES, N_GAIN) == (15, 8)


def test_yaml_equals_dataclass_defaults() -> None:
    cfg = load_kalmannet_config()
    assert cfg.net == KalmanNetSpec()
    assert cfg.train == TrainSpec()


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError):
        load_kalmannet_config(overrides={"net": {"hiden": 32}})


def test_invalid_values_rejected() -> None:
    with pytest.raises(ConfigError):
        KalmanNetSpec(hidden=0)
    with pytest.raises(ConfigError):
        TrainSpec(lr=0.0)
    with pytest.raises(ConfigError):
        TrainSpec(loss_weights=(0.0, 0.0, 0.0, 0.0))


def test_override_merges() -> None:
    cfg = load_kalmannet_config(overrides={"train": {"epochs": 3}})
    assert cfg.train.epochs == 3 and cfg.train.lr == 1e-3


def test_expected_num_params_default() -> None:
    assert expected_num_params(KalmanNetSpec()) == 30_664
