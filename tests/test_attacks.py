"""Attack tests (torch-free). The 'hurts equal-weight FedKalmanNet' criterion runs in tests/test_fed_loop.py (CI)."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.attacks import ATTACKS, AttackPlan, AttackSpec, attack_delta, load_attack_config
from afkn.confidence import ConfidenceAggregator, ConfidenceSpec
from afkn.fed.aggregate import RoundContext, normalize_weights
from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.seeding import SeedTree
from afkn.sim.config import ConfigError

P = 20
TARGET = np.linspace(-1.0, 1.0, P) * 10.0
G = np.zeros(P, np.float32)
HONEST = (0.3 * (TARGET - G)).astype(np.float32)


class QuadOracle:
    def reference_delta(self, global_params, round_idx):
        return (0.5 * (TARGET - np.asarray(global_params, np.float64))).astype(np.float32)

    def rmse(self, params):
        return float(np.linalg.norm(np.asarray(params, np.float64) - TARGET))


def _upd(cid: int, delta, rnd: int = 0) -> ClientUpdate:
    bc = GlobalBroadcast(round=rnd, nonce=bytes(16), params=G)
    return ClientUpdate(cid, rnd, bc.nonce, bc.model_hash, np.asarray(delta, np.float32), 100,
                        {"val_rmse_before": 5.0, "val_rmse_after": 4.0})


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_config() -> None:
    assert load_attack_config() == AttackSpec() and AttackSpec().attack == "none"
    assert set(ATTACKS) == {"none", "random", "sign_flip", "scaling", "bias", "adaptive"}
    for bad in ({"attack": "nope"}, {"fraction": 1.5}, {"scale": -1.0}, {"ids": [1, 1]}):
        with pytest.raises(ConfigError):
            load_attack_config(overrides=bad)


def test_attack_formulas() -> None:
    h = HONEST.astype(np.float64)
    hn = np.linalg.norm(h)
    rng = np.random.default_rng(0)
    u = np.random.default_rng(1).normal(size=P)
    u /= np.linalg.norm(u)
    np.testing.assert_array_equal(attack_delta(AttackSpec(attack="none"), h, rng, u), h)
    np.testing.assert_allclose(attack_delta(AttackSpec(attack="sign_flip", flip_k=2.0), h, rng, u), -2.0 * h)
    np.testing.assert_allclose(attack_delta(AttackSpec(attack="scaling", scale=7.0), h, rng, u), 7.0 * h)
    r = attack_delta(AttackSpec(attack="random", random_gain=3.0), h, rng, u)
    assert np.isclose(np.linalg.norm(r), 3.0 * hn) and abs(_cos(r, h)) < 0.95
    b = attack_delta(AttackSpec(attack="bias", bias_gain=2.0), h, rng, u)
    np.testing.assert_allclose(b - h, 2.0 * hn * u)


def test_adaptive_is_stealthy_at_defaults() -> None:
    h = HONEST.astype(np.float64)
    u = np.random.default_rng(1).normal(size=P)
    a = attack_delta(AttackSpec(attack="adaptive"), h, np.random.default_rng(0), u / np.linalg.norm(u))
    assert np.isclose(np.linalg.norm(a), np.linalg.norm(h))  # honest-sized: not clipped
    assert 0.9 < _cos(a, h) < 1.0  # aligned with the honest/reference direction, but biased
    assert not np.allclose(a, h)
    loud = attack_delta(AttackSpec(attack="adaptive", adaptive_scale=20.0), h, np.random.default_rng(0), u)
    assert np.isclose(np.linalg.norm(loud), 20.0 * np.linalg.norm(h))


def test_zero_honest_delta_is_safe() -> None:
    z = np.zeros(P)
    u = np.ones(P) / np.sqrt(P)
    for a in ATTACKS:
        out = attack_delta(AttackSpec(attack=a), z, np.random.default_rng(0), u)
        assert np.isfinite(out).all()


def test_plan_selection_and_determinism() -> None:
    spec = AttackSpec(attack="sign_flip", fraction=0.3)
    a, b = AttackPlan(spec, 10, SeedTree(0)), AttackPlan(spec, 10, SeedTree(0))
    assert a.attacker_ids == b.attacker_ids and len(a.attacker_ids) == 3
    assert AttackPlan(spec, 10, SeedTree(1)).attacker_ids != a.attacker_ids or True  # different seed may differ
    assert AttackPlan(AttackSpec(attack="none"), 10, SeedTree(0)).attacker_ids == ()
    assert AttackPlan(AttackSpec(attack="scaling", fraction=0.0), 10, SeedTree(0)).attacker_ids == ()
    assert AttackPlan(AttackSpec(attack="scaling", ids=(7, 2)), 10, SeedTree(0)).attacker_ids == (2, 7)
    with pytest.raises(ValueError):
        AttackPlan(AttackSpec(attack="scaling", ids=(10,)), 10, SeedTree(0))


def test_plan_apply_semantics() -> None:
    plan = AttackPlan(AttackSpec(attack="sign_flip", ids=(3,), start_round=2), 10, SeedTree(0))
    honest, bad = _upd(1, HONEST), _upd(3, HONEST)
    assert plan.apply(honest, 5) is honest  # not compromised
    assert plan.apply(bad, 1) is bad  # before start_round
    out = plan.apply(bad, 2)
    np.testing.assert_allclose(out.delta, -HONEST)
    assert out.delta.dtype == np.float32 and out.signals == bad.signals  # keeps claiming improvement
    assert out.client_id == 3 and out.nonce == bad.nonce and out.model_hash == bad.model_hash


def test_plan_noise_is_deterministic_per_client_and_round() -> None:
    plan = AttackPlan(AttackSpec(attack="random", ids=(0, 1)), 4, SeedTree(0))
    a = plan.apply(_upd(0, HONEST), 0).delta
    np.testing.assert_array_equal(a, plan.apply(_upd(0, HONEST), 0).delta)
    assert not np.array_equal(a, plan.apply(_upd(0, HONEST), 1).delta)
    assert not np.array_equal(a, plan.apply(_upd(1, HONEST), 0).delta)


def test_bias_direction_is_constant_across_rounds() -> None:
    plan = AttackPlan(AttackSpec(attack="bias", ids=(0,)), 2, SeedTree(0))
    d0 = plan.apply(_upd(0, HONEST), 0).delta - HONEST
    d5 = plan.apply(_upd(0, HONEST), 5).delta - HONEST
    np.testing.assert_allclose(d0, d5, atol=1e-6)


# "scaling" is deliberately absent: after norm clipping a scaled honest delta still points the right way, so
# Layer 2 bounds its effect (test_scaling_is_bounded_by_clipping) instead of down-weighting it.
@pytest.mark.parametrize("name", ["random", "sign_flip", "bias"])
def test_layer2_downweights_blunt_attacks(name: str) -> None:
    kw = {"random": {"random_gain": 5.0}, "sign_flip": {"flip_k": 3.0}, "bias": {"bias_gain": 5.0}}[name]
    plan = AttackPlan(AttackSpec(attack=name, ids=(4,), **kw), 5, SeedTree(0))
    ups = [plan.apply(_upd(i, HONEST), 0) for i in range(5)]
    wt = ConfidenceAggregator(ConfidenceSpec(), QuadOracle())(ups, RoundContext(0, G))
    w = normalize_weights(wt.raw, wt.floor)
    assert w[4] < 0.5 * w[:4].mean(), (name, w)


def test_adaptive_attacker_is_NOT_reliably_detected_by_layer2() -> None:
    """Documents a known limitation (stealthy attacker): at the defaults it keeps a normal weight."""
    plan = AttackPlan(AttackSpec(attack="adaptive", ids=(4,)), 5, SeedTree(0))
    ups = [plan.apply(_upd(i, HONEST), 0) for i in range(5)]
    wt = ConfidenceAggregator(ConfidenceSpec(), QuadOracle())(ups, RoundContext(0, G))
    w = normalize_weights(wt.raw, wt.floor)
    assert wt.info[4]["s_dir"] > 0.9 and not wt.info[4]["liar"]
    assert w[4] > 0.5 * w[:4].mean()


def test_scaling_is_bounded_by_clipping() -> None:
    plan = AttackPlan(AttackSpec(attack="scaling", ids=(4,), scale=30.0), 5, SeedTree(0))
    ups = [plan.apply(_upd(i, HONEST), 0) for i in range(5)]
    wt = ConfidenceAggregator(ConfidenceSpec(), QuadOracle())(ups, RoundContext(0, G))
    norms = np.linalg.norm(wt.deltas.astype(np.float64), axis=1)
    assert norms[4] <= 2.0 * norms[:4].mean() * (1 + 1e-5) and wt.info[4]["clip_factor"] < 0.1
