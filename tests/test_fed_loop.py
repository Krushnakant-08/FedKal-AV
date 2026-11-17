"""Federated loop (needs PyTorch). Locally skipped without torch; in CI (env CI set) must run."""

from __future__ import annotations

import os

import numpy as np
import pytest

if os.environ.get("CI"):
    import torch  # noqa: F401  hard import: CI must never skip these silently
else:
    pytest.importorskip("torch")

from afkn.data.partition import build_federated_dataset
from afkn.fed import FedSpec
from afkn.fed.loop import run_comparison, run_federated
from afkn.models import load_kalmannet_config

KCFG = load_kalmannet_config(
    overrides={"net": {"init_pos_gain": 0.3}, "train": {"batch_size": 6, "chunk_len": 50}}
)
FED = FedSpec(rounds=6, local_epochs=2, participation=0.6, eval_every=3)


@pytest.fixture(scope="module")
def dataset(small_cfg):
    return build_federated_dataset(small_cfg, seed=0)


def test_byte_accounting_and_logs(dataset) -> None:
    run = run_federated(dataset, KCFG, FedSpec(rounds=2, local_epochs=1, participation=0.6), seed=0)
    n_params = run.params.size
    assert n_params == 30_664
    for rl in run.rounds:
        k = len(rl.participants)
        assert k == 6
        assert rl.bytes_down == rl.bytes_up == k * n_params * 4  # project rule: params x 4 B x clients
        assert rl.wire_down > rl.bytes_down and rl.wire_up > rl.bytes_up
        assert [r["client"] for r in rl.clients] == rl.participants
        assert all(r["accepted"] and r["reject_reason"] is None for r in rl.clients)
        assert np.isclose(sum(r["weight"] for r in rl.clients), 1.0)
        for r in rl.clients:
            assert {"val_rmse_before", "val_rmse_after", "innov_ms", "train_loss"} <= set(r["signals"])
    assert run.total_bytes == sum(r.bytes_down + r.bytes_up for r in run.rounds)


def test_fedavg_weights_follow_n_samples(dataset) -> None:
    run = run_federated(dataset, KCFG, FedSpec(rounds=1, local_epochs=1, participation=1.0), seed=0)
    rows = run.rounds[0].clients
    total = sum(r["n_samples"] for r in rows)
    for r in rows:
        assert np.isclose(r["weight"], r["n_samples"] / total)


def test_federated_run_is_deterministic(dataset) -> None:
    f = FedSpec(rounds=2, local_epochs=1, participation=0.6)
    a = run_federated(dataset, KCFG, f, seed=3)
    b = run_federated(dataset, KCFG, f, seed=3)
    c = run_federated(dataset, KCFG, f, seed=4)
    np.testing.assert_array_equal(a.params, b.params)
    assert [r.participants for r in a.rounds] == [r.participants for r in b.rounds]
    assert not np.array_equal(a.params, c.params)


def test_global_model_changes_and_stays_finite(dataset) -> None:
    run = run_federated(dataset, KCFG, FedSpec(rounds=2, local_epochs=1), seed=0)
    assert np.isfinite(run.params).all() and np.isfinite(run.final_rmse)
    assert run.curve[-1][0] == 1  # last round is always evaluated


def test_federated_beats_local_only_and_gap_is_logged(dataset) -> None:
    """Done when: federated model beats local-only (global test); gap to centralized is logged."""
    out = run_comparison(dataset, KCFG, FED, seed=0)
    s = out["summary"]
    assert all(np.isfinite(s[k]) for k in ("fedavg_rmse", "local_only_mean_rmse", "centralized_rmse")), s
    assert np.isfinite(s["gap_fedavg_minus_centralized"])
    assert np.isclose(s["gap_fedavg_minus_centralized"], s["fedavg_rmse"] - s["centralized_rmse"])
    assert s["fedavg_beats_local_only"], (
        f"fedavg {s['fedavg_rmse']:.3f} m vs local-only mean {s['local_only_mean_rmse']:.3f} m "
        f"(centralized {s['centralized_rmse']:.3f} m)"
    )
