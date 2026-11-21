"""Layer 2 tests (torch-free, synthetic trusted-set oracle)."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.confidence import (
    ConfidenceAggregator,
    ConfidenceSpec,
    claim_flags,
    clip_to_median,
    combine_scores,
    direction_scores,
    load_confidence_config,
    perf_scores,
)
from afkn.fed.aggregate import RoundContext, normalize_weights
from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.sim.config import ConfigError

P = 20
TARGET = np.linspace(-1.0, 1.0, P) * 10.0  # the "ideal model" of the synthetic oracle


class QuadOracle:
    """rmse(p) = ||p - TARGET||; reference update points halfway to TARGET."""

    def reference_delta(self, global_params, round_idx):
        return (0.5 * (TARGET - np.asarray(global_params, np.float64))).astype(np.float32)

    def rmse(self, params):
        return float(np.linalg.norm(np.asarray(params, np.float64) - TARGET))


G = np.zeros(P, np.float32)
HONEST = (0.3 * (TARGET - G)).astype(np.float32)  # moves toward the target: measured gain > 0


def _upd(cid: int, delta, signals=None, n: int = 100, rnd: int = 0) -> ClientUpdate:
    bc = GlobalBroadcast(round=rnd, nonce=bytes(16), params=G)
    return ClientUpdate(cid, rnd, bc.nonce, bc.model_hash, np.asarray(delta, np.float32), n,
                        signals if signals is not None else {"val_rmse_before": 5.0, "val_rmse_after": 4.0})


def _agg(**kw) -> ConfidenceAggregator:
    return ConfidenceAggregator(ConfidenceSpec(**kw), QuadOracle())


def _run(agg, updates, rnd: int = 0):
    wt = agg(updates, RoundContext(rnd, G))
    w = normalize_weights(wt.raw, wt.floor)
    return wt, w


# ----------------------------------------------------------------------------- primitives
def test_clip_to_median() -> None:
    d = np.stack([np.ones(4), np.ones(4), 100 * np.ones(4)])
    out, f, med = clip_to_median(d, 2.0)
    assert np.isclose(med, 2.0)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), [2.0, 2.0, 4.0])  # capped at c x median = 4
    assert f[0] == f[1] == 1.0 and 0 < f[2] < 1
    z, fz, mz = clip_to_median(np.zeros((3, 4)), 2.0)
    assert mz == 0.0 and (fz == 1).all()


def test_direction_scores() -> None:
    ref = np.array([1.0, 0.0])
    d = np.array([[2.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(direction_scores(d, ref), [1.0, 0.0, 0.0, np.sqrt(0.5), 0.0], atol=1e-12)
    assert (direction_scores(d, np.zeros(2)) == 0).all()


def test_perf_scores() -> None:
    s = perf_scores(np.array([0.0, 1.0, -1.0, np.nan, -np.inf, np.inf]), 0.05)
    assert s[0] == 0.5 and s[1] > 0.99 and s[2] < 0.01 and s[3] < 1e-20 and s[4] < 1e-20 and s[5] >= 1 - 1e-12
    assert (np.diff(perf_scores(np.linspace(-1, 1, 50), 0.05)) > 0).all()


def test_claim_flags_truth_table() -> None:
    rep = np.array([1.0, 1.0, -1.0, 1.0, np.nan, 0.0])
    mea = np.array([-1.0, 1.0, -1.0, np.nan, 1.0, -1.0])
    np.testing.assert_array_equal(claim_flags(rep, mea, 0.05, -0.05), [True, False, False, True, True, False])


def test_combine_scores() -> None:
    a, b = np.array([0.5]), np.array([0.8])
    assert np.isclose(combine_scores(a, b, "product")[0], 0.4) and np.isclose(combine_scores(a, b, "mean")[0], 0.65)
    with pytest.raises(ValueError):
        combine_scores(a, b, "x")


def test_config() -> None:
    assert load_confidence_config() == ConfidenceSpec()
    assert ConfidenceSpec().weight_floor == 0.01 and ConfidenceSpec().clip_c == 2.0
    with pytest.raises(ConfigError):
        ConfidenceSpec(combine="max")
    with pytest.raises(ConfigError):
        load_confidence_config(overrides={"ema_alpha": 0.0})


# ----------------------------------------------------------------------------- aggregator
def test_weights_sum_to_one_floor_holds_and_everything_logged() -> None:
    agg = _agg()
    ups = [_upd(i, HONEST) for i in range(3)] + [_upd(3, -HONEST), _upd(4, np.zeros(P))]
    wt, w = _run(agg, ups)
    assert np.isclose(w.sum(), 1.0) and w.min() >= 0.01 - 1e-12
    for cid in range(5):
        row = wt.info[cid]
        assert {"clip_factor", "s_dir", "s_perf", "gain", "reported_gain", "liar", "score", "ema",
                "streak", "probation", "reason"} <= set(row)
    assert wt.info[3]["s_dir"] == 0.0 and w[3] < w[0] / 10  # sign-flip: near-floor weight
    assert wt.deltas.shape == (5, P)


def test_scaling_attack_has_bounded_effect() -> None:
    agg = _agg()
    ups = [_upd(i, HONEST) for i in range(4)] + [_upd(4, 1000.0 * HONEST)]
    wt, _ = _run(agg, ups)
    norms = np.linalg.norm(wt.deltas.astype(np.float64), axis=1)
    med = np.linalg.norm(HONEST.astype(np.float64))
    assert norms[4] <= 2.0 * med * (1 + 1e-5) and wt.info[4]["clip_factor"] < 0.01


def test_clip_off_ablation_leaves_deltas_untouched() -> None:
    wt, _ = _run(_agg(use_clip=False), [_upd(i, HONEST) for i in range(3)] + [_upd(3, 1000.0 * HONEST)])
    assert np.isclose(np.linalg.norm(wt.deltas[3]), 1000.0 * np.linalg.norm(HONEST), rtol=1e-4)
    assert wt.info[3]["clip_factor"] == 1.0


def test_liar_flag_halves_score() -> None:
    kw = {"use_direction": False}  # isolate the claim check
    bad = -HONEST  # measured degradation
    lie = {"val_rmse_before": 5.0, "val_rmse_after": 1.0}
    truth = {"val_rmse_before": 5.0, "val_rmse_after": 6.0}
    wt_lie, _ = _run(_agg(**kw), [_upd(0, HONEST), _upd(1, bad, lie)])
    wt_true, _ = _run(_agg(**kw), [_upd(0, HONEST), _upd(1, bad, truth)])
    assert wt_lie.info[1]["liar"] and not wt_true.info[1]["liar"]
    assert np.isclose(wt_lie.info[1]["score"], 0.5 * wt_true.info[1]["score"], atol=1e-12) or wt_true.info[1]["score"] < 1e-6
    assert not wt_lie.info[0]["liar"]


def test_missing_signals_are_flagged_not_crashing() -> None:
    wt, _ = _run(_agg(), [_upd(0, HONEST), _upd(1, HONEST, signals={})])
    assert wt.info[1]["liar"] and np.isnan(wt.info[1]["reported_gain"])


def test_ema_history() -> None:
    agg = _agg(ema_alpha=0.5)
    wt0, _ = _run(agg, [_upd(0, HONEST), _upd(1, HONEST)], rnd=0)
    s0 = wt0.info[0]["score"]
    assert wt0.info[0]["ema"] == s0
    wt1, _ = _run(agg, [_upd(0, 0.5 * HONEST), _upd(1, HONEST)], rnd=1)
    assert np.isclose(wt1.info[0]["ema"], 0.5 * s0 + 0.5 * wt1.info[0]["score"])
    # weights are proportional to n_i x ema_i
    assert np.isclose(wt1.raw[0], 100 * wt1.info[0]["ema"])


def test_history_off_uses_latest_score() -> None:
    agg = _agg(use_history=False)
    _run(agg, [_upd(0, HONEST)], rnd=0)
    wt, _ = _run(agg, [_upd(0, 0.5 * HONEST)], rnd=1)
    assert wt.info[0]["ema"] == wt.info[0]["score"]


def test_probation_cycle() -> None:
    agg = _agg()  # low_rounds=3, probation_len=5
    log = []
    for r in range(9):
        wt, w = _run(agg, [_upd(0, -HONEST), _upd(1, HONEST), _upd(2, HONEST), _upd(3, HONEST)], rnd=r)
        log.append((wt.info[0], w[0], wt.raw[0]))
    assert [bool(i["probation"]) for i, _, _ in log] == [False, False, True, True, True, True, True, False, False]
    assert log[2][0]["probation_started"] and log[7][0]["probation_ended"]
    for r in range(2, 7):
        info, weight, raw = log[r]
        assert raw == 0.0 and info["reason"] == "probation"
        assert np.isclose(weight, 0.01)  # floor still enforced, never silently discarded
    assert log[7][0]["ema"] == log[7][0]["score"]  # re-test: history reset


def test_good_round_resets_the_low_streak() -> None:
    agg = _agg()
    for r, delta in enumerate([-HONEST, -HONEST, HONEST, -HONEST, -HONEST]):
        wt, _ = _run(agg, [_upd(0, delta), _upd(1, HONEST), _upd(2, HONEST)], rnd=r)
        assert not wt.info[0]["probation"]


def test_non_participants_keep_their_state() -> None:
    agg = _agg()
    _run(agg, [_upd(0, -HONEST), _upd(1, HONEST), _upd(2, HONEST)], rnd=0)
    streak = agg.streak[0]
    _run(agg, [_upd(1, HONEST), _upd(2, HONEST)], rnd=1)
    assert agg.streak[0] == streak == 1


def test_direction_off_and_perf_off_ablations() -> None:
    wt, _ = _run(_agg(use_direction=False), [_upd(0, HONEST), _upd(1, -HONEST)])
    assert wt.info[1]["s_dir"] == 1.0
    wt, _ = _run(_agg(use_perf=False), [_upd(0, HONEST), _upd(1, -HONEST)])
    assert wt.info[1]["s_perf"] == 1.0


def test_honest_noisy_client_is_not_treated_like_an_attacker() -> None:
    """A noisy honest update (cos ~0.7) must keep a clearly larger weight than a sign-flip attacker."""
    rng = np.random.default_rng(0)
    noisy = HONEST + rng.normal(size=P).astype(np.float32) * np.linalg.norm(HONEST) / np.sqrt(P)
    wt, w = _run(_agg(), [_upd(0, HONEST), _upd(1, HONEST), _upd(2, noisy), _upd(3, -HONEST)])
    assert wt.info[2]["s_dir"] > 0.3 and w[2] > 5 * w[3]


def test_reported_perf_source_trusts_the_claim_and_never_queries_the_oracle() -> None:
    class NoOracle:
        def reference_delta(self, *a):
            raise AssertionError("oracle must not be used")

        def rmse(self, *a):
            raise AssertionError("oracle must not be used")

    agg = ConfidenceAggregator(ConfidenceSpec(perf_source="reported", use_direction=False, use_claim=False, use_clip=False), NoOracle())
    lie = {"val_rmse_before": 5.0, "val_rmse_after": 1.0}
    truth = {"val_rmse_before": 5.0, "val_rmse_after": 6.0}
    wt, _ = _run(agg, [_upd(0, HONEST, truth), _upd(1, -HONEST, lie)])
    assert wt.info[1]["s_perf"] > 0.99 > wt.info[0]["s_perf"]  # the forged claim wins: that is the weakness E8 measures
    with pytest.raises(ConfigError):
        ConfidenceSpec(perf_source="oracle")
