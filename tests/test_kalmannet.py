"""KalmanNet tests (need PyTorch). Locally they skip without torch; in CI (env CI set) they must run."""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

if os.environ.get("CI"):
    import torch  # hard import: CI must never skip these silently
else:
    torch = pytest.importorskip("torch")

from afkn.filters import EKF
from afkn.filters.consistency import position_rmse
from afkn.filters.params import matched_params
from afkn.models import expected_num_params, load_kalmannet_config
from afkn.models.kalmannet import KalmanNet
from afkn.models.motion_torch import f_torch
from afkn.models.train import evaluate, train
from afkn.seeding import SeedTree
from afkn.sim import motion
from afkn.sim.config import load_config
from afkn.sim.generator import generate_batch
from afkn.sim.sensors import realize_device


def _data(seed: int = 1, n: int = 8, steps: int = 100, mix=(0.5, 0.5, 0.0), sigma: float = 1.5):
    cfg = load_config(
        overrides={
            "steps": steps,
            "profiles": {
                "high_grade": {
                    "gnss": {"sigma": sigma, "bias_std": 0.0, "p_drop": 0.02},
                    "imu": {"sigma_yaw": 0.01, "sigma_acc": 0.1, "bias0_std_yaw": 0.0, "bias0_std_acc": 0.0,
                            "rw_yaw": 0.0, "rw_acc": 0.0, "scale_std": 0.0},
                }
            },
        }
    )
    dev = realize_device("high_grade", cfg.profiles["high_grade"], SeedTree(seed).child("dev").rng())
    return cfg, dev, generate_batch(cfg, dev, mix, n, SeedTree(seed).child("b"))


# ------------------------------------------------------------------ motion parity
@pytest.mark.parametrize("model", ["ctrv", "bicycle"])
def test_f_torch_matches_numpy(model: str) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 4)) * [100, 100, 1, 5] + [0, 0, 0, 15]
    u = rng.normal(size=(50, 2)) * [0.3, 1.0]
    u[:5, 0] = 0.0  # omega == 0 branch
    ref = np.stack([motion.f(x[i], u[i], 0.1, model) for i in range(50)])
    got = f_torch(torch.as_tensor(x), torch.as_tensor(u), 0.1, model).numpy()
    np.testing.assert_allclose(got, ref, atol=1e-10)


# ------------------------------------------------------------------ parameters
def test_num_params_and_roundtrip() -> None:
    net = KalmanNet(seed=0)
    assert net.num_params == expected_num_params(net.spec) == 30_664
    flat = net.get_flat_params()
    assert flat.dtype == np.float32 and flat.ndim == 1 and flat.size == net.num_params
    flat[:] += 1.0  # returned array is a copy
    assert not np.allclose(net.get_flat_params(), flat)
    other = KalmanNet(seed=1)
    other.set_flat_params(net.get_flat_params())
    np.testing.assert_array_equal(other.get_flat_params(), net.get_flat_params())


def test_set_flat_params_validates() -> None:
    net = KalmanNet(seed=0)
    n = net.num_params
    with pytest.raises(ValueError):
        net.set_flat_params(np.zeros(n - 1, dtype=np.float32))
    with pytest.raises(ValueError):
        net.set_flat_params(np.zeros((n, 1), dtype=np.float32))
    for bad in (np.nan, np.inf):
        arr = np.zeros(n, dtype=np.float32)
        arr[3] = bad
        with pytest.raises(ValueError):
            net.set_flat_params(arr)


def test_init_determinism() -> None:
    np.testing.assert_array_equal(KalmanNet(seed=3).get_flat_params(), KalmanNet(seed=3).get_flat_params())
    assert not np.array_equal(KalmanNet(seed=3).get_flat_params(), KalmanNet(seed=4).get_flat_params())


# ------------------------------------------------------------------ behaviour
def test_no_measurements_is_dead_reckoning() -> None:
    _, _, batch = _data()
    batch = dataclasses.replace(batch, z=np.full_like(batch.z, np.nan), z_mask=np.zeros_like(batch.z_mask))
    res = KalmanNet(seed=0).run(batch)
    x = batch.x0_est.copy()
    for t in range(batch.T):
        x = np.stack([motion.f(x[i], batch.u_imu[i, t], batch.dt, batch.model) for i in range(batch.n)])
        np.testing.assert_allclose(res.x_hat[:, t + 1], x, atol=1e-3)
    assert not res.updated.any()


def test_run_ignores_ground_truth() -> None:
    _, _, batch = _data()
    net = KalmanNet(seed=0)
    a = net.run(batch).x_hat
    scrambled = dataclasses.replace(
        batch, x=batch.x + 99.0, u_true=batch.u_true + 5.0, imu_bias=batch.imu_bias + 1.0
    )
    np.testing.assert_array_equal(a, net.run(scrambled).x_hat)


def test_run_output_shapes_and_finite() -> None:
    _, _, batch = _data()
    res = KalmanNet(seed=0).run(batch)
    assert res.x_hat.shape == batch.x.shape
    np.testing.assert_array_equal(res.x_hat[:, 0], batch.x0_est)
    assert np.isfinite(res.x_hat).all() and res.gain.shape == (batch.n, batch.T, 4, 2)


def test_gradients_finite_and_nonzero() -> None:
    _, _, batch = _data(n=4, steps=40)
    net = KalmanNet(seed=0)
    u = torch.as_tensor(batch.u_imu, dtype=torch.float32)
    z = torch.as_tensor(np.nan_to_num(batch.z), dtype=torch.float32)
    av = torch.as_tensor(batch.z_mask, dtype=torch.float32)
    xh, _ = net(u, z, av, x0_est=torch.as_tensor(batch.x0_est, dtype=torch.float32))
    loss = ((xh - torch.as_tensor(batch.x[:, 1:], dtype=torch.float32)) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in net.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0


def test_chunked_forward_equals_full() -> None:
    _, _, batch = _data(n=3, steps=60)
    net = KalmanNet(seed=0).eval()
    u = torch.as_tensor(batch.u_imu, dtype=torch.float32)
    z = torch.as_tensor(np.nan_to_num(batch.z), dtype=torch.float32)
    av = torch.as_tensor(batch.z_mask, dtype=torch.float32)
    x0 = torch.as_tensor(batch.x0_est, dtype=torch.float32)
    with torch.no_grad():
        full, _ = net(u, z, av, x0_est=x0)
        state, parts = net.init_state(x0), []
        for c in range(0, 60, 25):
            xh, state = net(u[:, c : c + 25], z[:, c : c + 25], av[:, c : c + 25], state=state.detach())
            parts.append(xh)
    np.testing.assert_allclose(torch.cat(parts, 1).numpy(), full.numpy(), atol=1e-5)


def test_init_pos_gain_pulls_toward_measurement() -> None:
    cfg = load_kalmannet_config(overrides={"net": {"init_pos_gain": 0.5}})
    _, _, batch = _data()
    res = KalmanNet(cfg, seed=0).run(batch)
    k = res.gain[:, :, [0, 1], [0, 1]]
    assert np.all(k > 0.3) and np.all(k < 0.7)


# ------------------------------------------------------------------ "Done when"
def test_overfits_single_trajectory() -> None:
    """Done when (a): training drives the loss on one trajectory down by >= 10x."""
    _, _, batch = _data(n=1, steps=100)
    cfg = load_kalmannet_config(overrides={"net": {"init_pos_gain": 0.3}, "train": {"epochs": 120, "batch_size": 1, "chunk_len": 50}})
    net = KalmanNet(cfg, seed=0)
    log = train(net, batch, cfg.train, seed=0)
    first, last = log.losses[0], float(np.mean(log.losses[-10:]))
    assert last * 10.0 <= first, f"loss went {first:.4g} -> {last:.4g}"


def test_beats_mismatched_ekf() -> None:
    """Done when (b): with a badly mis-tuned Q/R, a trained KalmanNet beats the EKF on held-out data."""
    cfg, dev, train_b = _data(seed=2, n=48, steps=150)
    _, _, test_b = _data(seed=3, n=24, steps=150)
    p = matched_params(cfg, dev)
    matched = position_rmse(EKF(p).run(test_b), test_b)
    mismatched = position_rmse(EKF(p.scaled(q=1e-4, r=100.0)).run(test_b), test_b)
    kcfg = load_kalmannet_config(overrides={"net": {"init_pos_gain": 0.3}, "train": {"epochs": 40}})
    net = KalmanNet(kcfg, seed=0)
    train(net, train_b, kcfg.train, seed=0)
    kn = evaluate(net, test_b)["position_rmse"]
    assert kn < mismatched, f"KalmanNet {kn:.3f} m vs mismatched EKF {mismatched:.3f} m (matched EKF {matched:.3f} m)"
