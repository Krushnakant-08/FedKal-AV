"""Step 2 filters: NEES/NIS consistency under matched noise, sensitivity to mismatch, UKF/ESKF behaviour.

Thresholds come from theory, not from tuning:
* ANEES per time step over N independent trajectories: N * ANEES ~ chi2(4 N). With N = 200 the
  per-step s.d. is sqrt(2*4/200) = 0.2, so the time-averaged value must sit well inside 4 +/- 0.3.
* Pooled NIS (innovations of a consistent filter are white): chi2(2), mean 2, s.d. of the mean
  ~ 2/sqrt(6e4) < 0.01, so +/- 0.05 is > 5 s.d.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.stats import chi2

from afkn.data import build_federated_dataset
from afkn.filters import (
    EKF,
    ESKF,
    UKF,
    ESKFSpec,
    FilterResult,
    UKFSpec,
    anees_bounds,
    consistency_summary,
    load_filter_config,
    matched_params,
    matched_params_by_profile,
    nees,
    position_rmse,
    state_error,
)
from afkn.filters.base import batch_cov
from afkn.filters.config import FilterConfig
from afkn.filters.ukf import sigma_points, ut_weights
from afkn.seeding import SeedTree
from afkn.sim.config import ConfigError, SplitSpec, load_config
from afkn.sim.generator import generate_batch
from afkn.sim.sensors import realize_device

# --------------------------------------------------------------------------- setups


# NOTE: gnss_sigma must differ from 1 so that R, sqrt(R) and R^2 are distinguishable by the tests.
def _overrides(*, steps=400, gnss_sigma=1.5, p_drop=0.02, bias_yaw=0.0, bias_acc=0.0, model="ctrv", profile="high_grade"):
    return {
        "steps": steps,
        "motion_model": model,
        "profiles": {
            profile: {
                "gnss": {"sigma": gnss_sigma, "bias_std": 0.0, "p_drop": p_drop},
                "imu": {"sigma_yaw": 0.01, "sigma_acc": 0.1, "bias0_std_yaw": bias_yaw, "bias0_std_acc": bias_acc,
                        "rw_yaw": 0.0, "rw_acc": 0.0, "scale_std": 0.0},
            }
        },
    }


def _setup(seed=1, n=200, mix=(0.4, 0.4, 0.2), profile="high_grade", **kw):
    cfg = load_config(overrides=_overrides(profile=profile, **kw))
    dev = realize_device(profile, cfg.profiles[profile], SeedTree(seed).child("dev").rng())
    batch = generate_batch(cfg, dev, mix, n, SeedTree(seed).child("b"))
    return cfg, dev, batch, matched_params(cfg, dev)


@pytest.fixture(scope="module")
def matched():
    """Data whose only errors are the ones the filters model (no IMU bias/RW/scale, no GNSS bias)."""
    return _setup()


@pytest.fixture(scope="module")
def results(matched):
    _, _, batch, p = matched
    return {"ekf": EKF(p).run(batch), "ukf": UKF(p).run(batch)}


@pytest.fixture(scope="module")
def biased():
    """Mid-grade-like device with a large constant IMU bias and tunnel-heavy routes."""
    return _setup(seed=2, n=100, mix=(0.3, 0.3, 0.4), profile="mid_grade", gnss_sigma=1.5, steps=600, bias_yaw=0.05, bias_acc=0.3)


# --------------------------------------------------------- consistency (Done-when for Step 2)


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_anees_matches_state_dimension_under_matched_noise(matched, results, name):
    _, _, batch, _ = matched
    s = consistency_summary(results[name], batch)
    lo, hi = s["anees_bounds"]
    assert 3.7 < s["anees_mean"] < 4.3, s
    assert lo < s["anees_mean"] < hi
    assert s["anees_frac_in_bounds"] >= 0.85, s  # ~0.95 expected for a consistent filter


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_nis_consistent_under_matched_noise(matched, results, name):
    _, _, batch, _ = matched
    r = results[name]
    s = consistency_summary(r, batch)
    assert s["n_nis"] > 50_000
    assert abs(s["nis_mean"] - 2.0) < 0.05, s
    assert abs(s["nis_coverage"] - 0.95) < 0.02, s


def test_innovations_are_white_under_matched_noise(matched, results):
    """A consistent filter has uncorrelated normalised innovations (lag-1 correlation ~ 0)."""
    r = results["ekf"]
    L = np.linalg.cholesky(np.where(np.isnan(r.S), np.eye(2), r.S))
    v = np.linalg.solve(L, np.where(np.isnan(r.innov), 0.0, r.innov)[..., None])[..., 0]  # (N,T,2)
    both = r.updated[:, 1:] & r.updated[:, :-1]
    for c in (0, 1):
        a, b = v[:, 1:, c][both], v[:, :-1, c][both]
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.02  # s.e. ~ 1/sqrt(6e4) = 0.004


@pytest.mark.parametrize("model", ["ctrv", "bicycle"])
def test_ekf_consistent_for_both_motion_models(model):
    _, _, batch, p = _setup(model=model, n=150, seed=5)
    s = consistency_summary(EKF(p).run(batch), batch)
    assert 3.65 < s["anees_mean"] < 4.35 and abs(s["nis_mean"] - 2.0) < 0.06, s


def test_nees_and_nis_detect_mismatched_Q_and_R(matched):
    """The consistency tests have teeth: wrong Q or R is flagged, in the expected direction."""
    _, _, batch, p = matched
    lo, hi = anees_bounds(batch.n)
    under_q = consistency_summary(EKF(p.scaled(q=0.01)).run(batch), batch)
    over_q = consistency_summary(EKF(p.scaled(q=100.0)).run(batch), batch)
    under_r = consistency_summary(EKF(p.scaled(r=0.01)).run(batch), batch)
    over_r = consistency_summary(EKF(p.scaled(r=100.0)).run(batch), batch)
    assert under_q["anees_mean"] > 10 * hi  # overconfident
    assert over_q["anees_mean"] < lo  # underconfident
    assert under_r["nis_mean"] > 20.0 and over_r["nis_mean"] < 0.5
    assert under_q["anees_frac_in_bounds"] < 0.1 and over_q["anees_frac_in_bounds"] < 0.1


# --------------------------------------------------------------------- structural


@pytest.mark.parametrize("name", ["ekf", "ukf"])
def test_covariances_symmetric_psd_and_updates_reduce_uncertainty(matched, results, name):
    r = results[name]
    for P in (r.P, r.P_pred):
        assert np.abs(P - np.swapaxes(P, -1, -2)).max() < 1e-9
        assert np.linalg.eigvalsh(P).min() > -1e-9
    tr_post = np.trace(r.P[:, 1:], axis1=-2, axis2=-1)
    tr_pred = np.trace(r.P_pred, axis1=-2, axis2=-1)
    assert (tr_post[r.updated] < tr_pred[r.updated]).all()
    assert np.array_equal(r.P[:, 1:][~r.updated], r.P_pred[~r.updated])  # no update -> posterior == prior


def test_outages_are_pure_dead_reckoning(matched):
    cfg, dev, _, p = matched
    batch = generate_batch(cfg, dev, (0.0, 0.0, 1.0), 20, SeedTree(9).child("b"))  # tunnel only
    for flt in (EKF(p), UKF(p), ESKF(p)):
        r = flt.run(batch)
        assert not r.updated.any() and np.isnan(r.innov).all() and np.isnan(r.nis).all()
        assert np.array_equal(r.x_hat[:, 1:], r.x_pred)
        tr = np.trace(r.P, axis1=-2, axis2=-1)
        assert (np.diff(tr, axis=1) > 0).all()  # uncertainty grows monotonically without GNSS


def test_filters_use_only_u_imu_z_mask_and_x0_est(matched):
    cfg, dev, batch, p = matched
    sub = batch.subset(slice(0, 10))
    scrambled = dataclasses.replace(
        sub, x=np.full_like(sub.x, np.nan), u_true=np.full_like(sub.u_true, np.nan),
        imu_bias=np.full_like(sub.imu_bias, np.nan),
    )
    for flt in (EKF(p), UKF(p), ESKF(p)):
        a, b = flt.run(sub), flt.run(scrambled)
        assert np.array_equal(a.x_hat, b.x_hat) and np.array_equal(a.P, b.P)


def test_runs_are_deterministic(matched):
    _, _, batch, p = matched
    sub = batch.subset(slice(0, 20))
    for flt in (EKF(p), UKF(p), ESKF(p)):
        a, b = flt.run(sub), flt.run(sub)
        assert np.array_equal(a.x_hat, b.x_hat) and np.array_equal(a.P, b.P)


def test_noise_free_system_is_tracked_almost_exactly():
    ov = _overrides(steps=200, gnss_sigma=1e-6, p_drop=0.0)
    ov["process_noise"] = {"pos": 0.0, "heading": 0.0, "speed": 0.0}
    ov["init_noise"] = {"pos": 0.0, "heading": 0.0, "speed": 0.0}
    ov["profiles"]["high_grade"]["imu"].update({"sigma_yaw": 1e-9, "sigma_acc": 1e-9})
    cfg = load_config(overrides=ov)
    dev = realize_device("high_grade", cfg.profiles["high_grade"], SeedTree(0).rng())
    batch = generate_batch(cfg, dev, (0.5, 0.5, 0.0), 10, SeedTree(0).child("b"))
    r = EKF(matched_params(cfg, dev)).run(batch)
    assert position_rmse(r, batch) < 1e-4


# ----------------------------------------------------------------------------- UKF


@pytest.mark.parametrize("alpha, beta, kappa", [(1.0, 2.0, 0.0), (0.5, 2.0, 0.0), (0.7, 2.0, 1.0)])
def test_sigma_points_reproduce_mean_and_covariance(alpha, beta, kappa):
    rng = np.random.default_rng(0)
    d = 6
    A = rng.normal(size=(3, d, d))
    P = A @ np.swapaxes(A, -1, -2) + np.eye(d)
    mean = rng.normal(size=(3, d))
    X, wm, wc = sigma_points(mean, np.linalg.cholesky(P), alpha, beta, kappa)
    assert X.shape == (3, 2 * d + 1, d) and np.isclose(wm.sum(), 1.0)
    assert np.allclose(np.einsum("k,nkd->nd", wm, X), mean)
    dX = X - mean[:, None]
    assert np.allclose(np.einsum("k,nki,nkj->nij", wc, dX, dX), P)


def test_unscented_transform_is_exact_for_linear_maps():
    rng = np.random.default_rng(1)
    d = 4
    A = rng.normal(size=(d, d))
    P = (lambda M: M @ M.T + np.eye(d))(rng.normal(size=(d, d)))[None]
    m = rng.normal(size=(1, d))
    X, wm, wc = sigma_points(m, np.linalg.cholesky(P), 1.0, 2.0, 0.0)
    Y = X @ A.T
    ym = np.einsum("k,nkd->nd", wm, Y)
    dY = Y - ym[:, None]
    assert np.allclose(ym, m @ A.T)
    assert np.allclose(np.einsum("k,nki,nkj->nij", wc, dY, dY), A @ P[0] @ A.T)
    assert np.allclose(ut_weights(4, 1.0, 2.0, 0.0)[0][0], 0.0)  # alpha=1, kappa=0: zero centre weight


def test_ukf_tracks_like_ekf_when_noise_is_matched(matched, results):
    _, _, batch, _ = matched
    r_ekf, r_ukf = position_rmse(results["ekf"], batch), position_rmse(results["ukf"], batch)
    assert abs(r_ukf / r_ekf - 1.0) < 0.02
    # Same filter, different linearisation: the estimates must never differ by more than a small
    # fraction of the filter's own posterior position std (measured max ~0.08 on this data).
    d = np.linalg.norm(results["ukf"].x_hat[..., :2] - results["ekf"].x_hat[..., :2], axis=-1)
    sd = np.sqrt(np.trace(results["ekf"].P[..., :2, :2], axis1=-2, axis2=-1))
    assert (d / sd).max() < 0.25


# ---------------------------------------------------------------------- simplified ESKF


def test_eskf_without_bias_states_equals_ekf(matched):
    _, _, batch, p = matched
    a = EKF(p).run(batch)
    b = ESKF(p, ESKFSpec(estimate_bias=False)).run(batch)
    assert b.bias_hat is None
    for f in ("x_hat", "P", "x_pred", "P_pred"):
        assert np.allclose(getattr(a, f), getattr(b, f), atol=1e-8), f
    assert np.allclose(a.innov, b.innov, atol=1e-8, equal_nan=True)
    assert np.allclose(a.S, b.S, atol=1e-8, equal_nan=True)


def test_eskf_bias_states_help_when_imu_has_a_real_bias(biased):
    _, dev, batch, p = biased
    ekf, eskf = EKF(p).run(batch), ESKF(p, load_filter_config().eskf).run(batch)
    assert max(abs(v) for v in dev.imu_bias0) > 0.02  # the scenario really has a bias
    assert position_rmse(eskf, batch) < 0.95 * position_rmse(ekf, batch)
    est = eskf.bias_hat[:, -1].mean(axis=0)
    # the yaw-rate bias estimate moved most of the way from the zero prior towards the truth
    assert abs(est[0] - dev.imu_bias0[0]) < 0.5 * abs(dev.imu_bias0[0])


def test_eskf_cost_without_any_bias_is_bounded(matched):
    """Honest check of the downside: extra bias states cost some accuracy when there is no bias."""
    _, _, batch, p = matched
    eskf = ESKF(p, load_filter_config().eskf).run(batch)
    assert position_rmse(eskf, batch) < 1.3 * position_rmse(EKF(p).run(batch), batch)


# --------------------------------------------------------------- params / helpers / config


def test_matched_params_equal_simulator_noise(matched):
    cfg, dev, _, p = matched
    assert np.array_equal(p.Q, cfg.Q()) and np.array_equal(p.P0, cfg.P0())
    assert np.allclose(p.R, dev.R()) and np.allclose(p.Qu, np.diag(np.square(dev.imu_sigma)))
    assert p.model == cfg.motion_model and p.dt == cfg.dt
    q = p.scaled(q=3.0, r=5.0)
    assert np.allclose(q.Q, 3 * p.Q) and np.allclose(q.R, 5 * p.R) and np.allclose(q.Qu, 3 * p.Qu)
    assert np.allclose(p.scaled(qu=7.0).Qu, 7 * p.Qu)


def test_mixed_sets_use_per_trajectory_R_correctly(default_cfg):
    cfg = default_cfg.replace(steps=120, splits=SplitSpec(train=2, val=1, test=1, global_test=10, trusted=10))
    trusted = build_federated_dataset(cfg).trusted
    p = matched_params_by_profile(cfg, trusted.profile_idx)
    assert p.R.shape == (10, 2, 2)
    for i, name in enumerate(cfg.profile_names):
        sel = trusted.profile_idx == i
        assert np.allclose(p.R[sel], cfg.profiles[name].gnss.sigma ** 2 * np.eye(2))
    full = EKF(p).run(trusted)
    for i, name in enumerate(cfg.profile_names):  # same as filtering each profile separately with shared R
        sel = np.flatnonzero(trusted.profile_idx == i)
        sub = trusted.subset(sel)
        shared = dataclasses.replace(p, R=p.R[sel[0]], Qu=p.Qu[sel[0]])
        assert np.allclose(EKF(shared).run(sub).x_hat, full.x_hat[sel], atol=1e-9), name


def test_input_validation(matched):
    _, _, batch, p = matched
    with pytest.raises(ValueError, match="dt"):
        EKF(dataclasses.replace(p, dt=0.2)).run(batch)
    with pytest.raises(ValueError, match="R"):
        EKF(dataclasses.replace(p, R=np.stack([p.R] * 3))).run(batch)
    with pytest.raises(ValueError):
        dataclasses.replace(p, model="segway")
    with pytest.raises(ValueError):
        dataclasses.replace(p, Q=np.eye(3))
    with pytest.raises(ValueError):
        batch_cov(np.eye(2), 5, "x")[0] and batch_cov(np.zeros((4, 2, 2)), 5, "x")


def test_nees_helpers_on_hand_computed_cases():
    lo, hi = anees_bounds(1, 4, 0.95)
    assert np.isclose(lo, chi2.ppf(0.025, 4)) and np.isclose(hi, chi2.ppf(0.975, 4))
    assert anees_bounds(100)[1] - anees_bounds(100)[0] < anees_bounds(10)[1] - anees_bounds(10)[0]
    # heading error wraps: 3.0 vs -3.0 is 6.0 - 2*pi, not 6.0
    e = state_error(np.array([[0.0, 0.0, 3.0, 0.0]]), np.array([[0.0, 0.0, -3.0, 0.0]]))
    assert np.isclose(e[0, 2], 6.0 - 2 * np.pi)
    # NEES with P = I and a unit position error is exactly 1; with P = 4 I it is 0.25
    n, T = 1, 1
    true = type("B", (), {"x": np.zeros((n, T + 1, 4))})()
    xh = np.zeros((n, T + 1, 4))
    xh[..., 0] = 1.0
    for scale, want in ((1.0, 1.0), (4.0, 0.25)):
        P = np.broadcast_to(scale * np.eye(4), (n, T + 1, 4, 4)).copy()
        res = FilterResult(xh, P, np.zeros((n, T, 4)), np.zeros((n, T, 4, 4)), np.zeros((n, T, 2)),
                           np.zeros((n, T, 2, 2)), np.zeros((n, T), bool))
        assert np.allclose(nees(res, true), want)


def test_filter_config_defaults_and_validation():
    fc = load_filter_config()
    assert isinstance(fc, FilterConfig) and fc.ukf == UKFSpec() and fc.eskf == ESKFSpec()
    assert load_filter_config(overrides={"eskf": {"estimate_bias": False}}).eskf.estimate_bias is False
    with pytest.raises(ConfigError):
        load_filter_config(overrides={"ukf": {"alpha": 0.0}})
    with pytest.raises(ConfigError):
        load_filter_config(overrides={"ukf": {"gamma": 1.0}})
    with pytest.raises(ConfigError):
        load_filter_config(overrides={"eskf": {"bias_init_std": [-1.0, 0.2]}})
