"""Aggregation primitives and config (torch-free)."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.fed import (
    FedSpec,
    apply_deltas,
    fedavg_weights,
    load_fed_config,
    normalize_weights,
    sample_participants,
)
from afkn.sim.config import ConfigError


def test_fedavg_weights() -> None:
    w = fedavg_weights([100, 300])
    np.testing.assert_allclose(w, [0.25, 0.75])
    with pytest.raises(ValueError):
        fedavg_weights([0, 0])


def test_normalize_weights_sum_and_floor() -> None:
    w = normalize_weights([1.0, 0.0, 0.0, 9.0], floor=0.01)
    assert np.isclose(w.sum(), 1.0) and w.min() >= 0.01 - 1e-15
    assert np.isclose(w[1], 0.01)  # a zero-score client sits exactly on the floor, not below
    assert w[3] > w[0] > w[1]


def test_normalize_weights_edge_cases() -> None:
    np.testing.assert_allclose(normalize_weights([0.0, 0.0, 0.0, 0.0], 0.01), 0.25)
    np.testing.assert_allclose(normalize_weights([2.0, 6.0]), [0.25, 0.75])
    for bad in ([], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0]):
        with pytest.raises(ValueError):
            normalize_weights(bad)
    with pytest.raises(ValueError):
        normalize_weights([1.0, 1.0], floor=0.6)  # 2 * 0.6 > 1


def test_sample_participants() -> None:
    a = sample_participants(10, 0.6, np.random.default_rng(1))
    b = sample_participants(10, 0.6, np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 6 and len(set(a.tolist())) == 6 and (np.diff(a) > 0).all()
    assert len(sample_participants(10, 0.01, np.random.default_rng(0))) == 1
    assert len(sample_participants(10, 1.0, np.random.default_rng(0))) == 10
    with pytest.raises(ValueError):
        sample_participants(10, 0.0, np.random.default_rng(0))


def test_apply_deltas_matches_manual() -> None:
    g = np.arange(5, dtype=np.float32)
    d = np.array([[1, 0, 0, 0, 0], [0, 2, 0, 0, 0]], dtype=np.float32)
    out = apply_deltas(g, d, np.array([0.25, 0.75]), server_lr=2.0)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, g + 2.0 * np.array([0.25, 1.5, 0, 0, 0]))


def test_apply_deltas_rejects_bad_input() -> None:
    g = np.zeros(3, np.float32)
    ok = np.ones((2, 3), np.float32)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError):
        apply_deltas(g, np.ones((2, 4), np.float32), w)
    with pytest.raises(ValueError):
        apply_deltas(g, ok, np.array([0.5, 0.6]))
    with pytest.raises(ValueError):
        apply_deltas(g, ok, np.array([-0.5, 1.5]))
    bad = ok.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        apply_deltas(g, bad, w)


def test_fed_config_defaults_and_validation() -> None:
    cfg = load_fed_config()
    assert cfg == FedSpec()
    assert (cfg.rounds, cfg.local_epochs, cfg.participation) == (100, 2, 0.6)
    assert load_fed_config(overrides={"rounds": 3}).rounds == 3
    with pytest.raises(ConfigError):
        load_fed_config(overrides={"rounds": 0})
    with pytest.raises(ConfigError):
        load_fed_config(overrides={"participation": 1.5})
    with pytest.raises(ConfigError):
        load_fed_config(overrides={"nope": 1})
