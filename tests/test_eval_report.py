"""Torch-free tests for the evaluation layer: config, cells, statistics, fairness audit, report generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from afkn.eval.cells import (
    METHODS,
    REPORTED_ONLY,
    VARIANTS,
    Cell,
    build_grid,
    build_sim_config,
    merge_exps,
)
from afkn.eval.config import FairnessSpec, load_eval_config
from afkn.eval.fairness import audit
from afkn.eval.metrics import rounds_to_converge
from afkn.eval.report import flatten_rounds, generate_report, records_frame
from afkn.eval.stats import mean_std, paired_wilcoxon
from afkn.sim.config import ConfigError, load_config


# ------------------------------------------------------------------ config and cells
def test_scales_load_and_validate() -> None:
    for s in ("smoke", "small", "full"):
        assert load_eval_config(s).scale.seeds
    assert len(load_eval_config("full").scale.seeds) >= 5 and load_eval_config("full").scale.rounds == 100
    with pytest.raises(ConfigError):
        load_eval_config("nope")
    with pytest.raises(ConfigError):
        load_eval_config("smoke", overrides={"bogus": 1})
    with pytest.raises(ConfigError):
        load_eval_config("smoke", overrides={"hetero_levels": ["wild"]})
    assert load_eval_config("smoke", overrides={"seeds": [3]}).scale.seeds == (3,)


def test_grid_is_deterministic_deduplicated_and_complete() -> None:
    sc = load_eval_config("smoke").scale
    g1, g2 = build_grid(sc), build_grid(sc)
    assert g1 == g2
    merged = merge_exps(g1)
    assert len(merged) < len(g1)  # E1/E2/E6 share cells
    shared = [c for c, tags in merged.values() if {"E1", "E2"} <= set(tags)]
    assert shared and all(c.method == "fedavg" for c in shared)
    assert {c.method for c, _ in merged.values()} == set(METHODS)
    ids = [c.id for c, _ in merged.values()]
    assert len(set(ids)) == len(ids)
    assert all(c.attack_frac == 0.0 for c, _ in merged.values() if c.attack == "none")


def test_cell_id_ignores_experiment_tag_only() -> None:
    a, b = Cell("E1", "fedavg", 0), Cell("E2", "fedavg", 0)
    assert a.id == b.id and a.id != Cell("E1", "fedavg", 1).id and a.id != Cell("E1", "proposed", 0).id
    from afkn.eval.cells import _cell

    with pytest.raises(ValueError):
        _cell("E1", "nope", 0)


def test_sim_config_for_cells() -> None:
    base = load_config()
    sc = load_eval_config("smoke").scale
    homog = build_sim_config(base, Cell("E1", "fedavg", 0, hetero="homog"), sc)
    assert {c.profile for c in homog.clients} == {"mid_grade"} and homog.steps == sc.steps
    mild = build_sim_config(base, Cell("E1", "fedavg", 0, hetero="mild"), sc)
    assert {c.profile for c in mild.clients} == {"high_grade", "mid_grade", "low_grade"}
    strong = build_sim_config(base, Cell("E1", "fedavg", 0), sc)
    assert [c.profile for c in strong.clients] == [c.profile for c in base.clients]
    noisy = build_sim_config(base, Cell("E1", "fedavg", 0, noise=2.0), sc)
    assert all(np.isclose(n.sigma_scale, 2.0 * b.sigma_scale) for n, b in zip(noisy.clients, base.clients, strict=True))
    for frac, want in [(0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3)]:
        cfg = build_sim_config(base, Cell("E3", "fedavg", 0, faulty_frac=frac), sc)
        assert sum(c.profile == "faulty" for c in cfg.clients) == want
    cfg3 = build_sim_config(base, Cell("E3", "fedavg", 0, faulty_frac=0.3), sc)
    assert {c.fault_mode for c in cfg3.clients if c.profile == "faulty"} == {"constant_bias", "random_spikes"}


def test_variants_and_reported_overrides_are_valid_confidence_specs() -> None:
    from afkn.confidence import ConfidenceSpec

    for over in [*VARIANTS.values(), REPORTED_ONLY]:
        ConfidenceSpec(**over)


# ------------------------------------------------------------------ statistics
def test_mean_std() -> None:
    m, s, n = mean_std([1.0, 3.0, np.nan])
    assert (m, n) == (2.0, 2) and np.isclose(s, np.sqrt(2.0))
    assert mean_std([5.0]) == (5.0, 0.0, 1)
    assert np.isnan(mean_std([np.nan])[0])


def test_paired_wilcoxon_known_cases() -> None:
    a = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    res = paired_wilcoxon(a, a - 1.0)  # a larger everywhere
    assert res["n"] == res["n_nonzero"] == 8 and res["effect_r"] == 1.0 and res["median_diff"] == 1.0
    assert res["p"] < 0.05
    rev = paired_wilcoxon(a - 1.0, a)
    assert rev["effect_r"] == -1.0
    same = paired_wilcoxon(a, a)
    assert np.isnan(same["p"]) and same["n_nonzero"] == 0
    few = paired_wilcoxon([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert few["p"] >= 0.25 and few["effect_r"] == 1.0  # exact two-sided p for n = 3 cannot reach 0.05
    with pytest.raises(ValueError):
        paired_wilcoxon([1.0], [1.0, 2.0])


def test_wilcoxon_ignores_nonfinite_pairs() -> None:
    res = paired_wilcoxon([1.0, np.nan, 3.0, 4.0], [0.0, 1.0, 2.0, 2.0])
    assert res["n"] == 3


def test_rounds_to_converge() -> None:
    curve = [(0, 10.0), (1, 6.0), (2, 5.1), (3, 5.0)]
    assert rounds_to_converge(curve, 0.05) == 2
    assert rounds_to_converge([], 0.05) is None and rounds_to_converge([(0, float("nan"))]) is None


# ------------------------------------------------------------------ fairness audit
def _rows(spec: list[tuple[str, float, bool]], k: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{"profile": p, "weight": w, "accepted": True, "attacker": a, "k_accepted": k, "method": "m"}
                         for p, w, a in spec])


def test_audit_flags_penalised_low_grade_and_reports_attackers() -> None:
    fair = _rows([("high_grade", 0.25, False), ("mid_grade", 0.25, False), ("low_grade", 0.25, False), ("faulty", 0.25, False)])
    a = audit(fair, FairnessSpec())
    assert np.isclose(a["ratio_low_vs_reference"], 1.0) and not a["concern"] and np.isnan(a["attacker_relative_weight"])
    unfair = _rows([("high_grade", 0.3, False), ("mid_grade", 0.3, False), ("low_grade", 0.05, False), ("faulty", 0.01, True)])
    b = audit(unfair, FairnessSpec())
    assert b["ratio_low_vs_reference"] < 0.5 and b["concern"]
    assert np.isclose(b["attacker_relative_weight"], 0.05)  # 0.01 x k=5
    assert audit(pd.DataFrame(), FairnessSpec())["concern"] is False


def test_audit_ignores_rejected_updates() -> None:
    d = _rows([("high_grade", 0.5, False), ("low_grade", 0.5, False)])
    d.loc[1, "accepted"] = False
    assert np.isnan(audit(d, FairnessSpec())["ratio_low_vs_reference"])


# ------------------------------------------------------------------ report generation from synthetic records
def _record(method: str, seed: int, exps: list[str], rmse: float, *, attack: str = "none", frac: float = 0.0,
            variant: str = "default", faulty_frac=None, participation: float = 0.6, with_rounds: bool = True) -> dict:
    cell = {"exp": exps[0], "method": method, "seed": seed, "noise": 1.0, "hetero": "strong", "faulty_frac": faulty_frac,
            "participation": participation, "attack": attack, "attack_frac": frac, "variant": variant}
    rounds = []
    if with_rounds and method in ("fedavg", "proposed", "proposed_reported"):
        for r in range(2):
            clients = []
            for cid, prof in enumerate(["high_grade", "mid_grade", "low_grade", "faulty"]):
                atk = attack != "none" and cid == 3
                w = 0.25 if method == "fedavg" else (0.05 if atk else 0.3)
                clients.append({"client": cid, "profile": prof, "accepted": True, "reject_reason": None, "weight": w,
                                "attacker": atk, "n_samples": 10, "delta_norm": 1.0, "bytes_up": 100, "signals": {}})
            rounds.append({"round": r, "participants": [0, 1, 2, 3], "bytes_down": 400, "bytes_up": 400,
                           "global_rmse": rmse, "clients": clients})
    return {"id": f"{method}-{seed}-{attack}-{frac}-{variant}-{faulty_frac}-{participation}-{exps}", "cell": cell,
            "method": method, "exps": exps, "final_rmse": rmse, "curve": [[0, rmse * 1.5], [1, rmse]],
            "total_bytes": 800 * participation if rounds else 0, "rounds": rounds, "attackers": [3] if attack != "none" else [],
            "profiles": ["high_grade", "mid_grade", "low_grade", "faulty"], "client_test_rmse": [rmse] * 4,
            "per_profile_rmse": {"high_grade": rmse}, "elapsed_s": 1.0}


def _synthetic_records() -> list[dict]:
    recs = []
    for seed in range(6):
        for m, base in [("fedavg", 6.0), ("proposed", 5.5), ("local_only", 8.0), ("centralized", 5.0), ("ekf", 7.0)]:
            recs.append(_record(m, seed, ["E1", "E2", "E6"] if m == "fedavg" else ["E2", "E6"], base + 0.1 * seed))
        for ff in (0.0, 0.3):
            for m in ("fedavg", "proposed"):
                recs.append(_record(m, seed, ["E3"], 6.0 + ff, faulty_frac=ff))
        for p in (0.6, 1.0):
            for m in ("fedavg", "proposed"):
                recs.append(_record(m, seed, ["E4"], 6.0, participation=p))
        for v in ("default", "no_direction"):
            for a in ("none", "sign_flip"):
                recs.append(_record("proposed", seed, ["E5"], 6.0 + (a != "none"), attack=a, frac=0.2 if a != "none" else 0.0, variant=v))
        for m in ("fedavg", "proposed_reported", "proposed"):
            recs.append(_record(m, seed, ["E8"], 6.0))
            recs.append(_record(m, seed, ["E8"], 9.0 if m == "fedavg" else 6.5, attack="sign_flip", frac=0.3))
    return recs


def test_report_generates_all_tables_figures_and_summary(tmp_path) -> None:
    ec = load_eval_config("smoke")
    files = generate_report(_synthetic_records(), tmp_path, ec)
    names = {f.name for f in files}
    for want in ("E1_heterogeneity_fedavg.csv", "E2_methods.csv", "E2_stats_proposed_vs_fedavg.csv", "E3_faulty_clients.csv",
                 "E4_tradeoffs.csv", "E5_ablations.csv", "E6_fairness_weights.csv", "E6_error_by_client_class.csv",
                 "E8_rmse.csv", "E8_degradation_vs_no_attack.csv", "E8_attacker_weights.csv", "E8_stats.csv", "summary.md",
                 "E2_convergence.png", "E3_faulty_clients.png", "E4_accuracy_vs_mb.png", "E5_ablations.png",
                 "E6_weights_by_profile.png", "E8_degradation.png"):
        assert want in names, want
    assert all(f.exists() and f.stat().st_size > 0 for f in files)
    summary = (tmp_path / "summary.md").read_text()
    assert "E7" in summary and "Not run" in summary and "nothing is quoted from papers" in summary
    e2 = pd.read_csv(tmp_path / "tables" / "E2_methods.csv")
    m = e2[e2["method"] == "fedavg"]["mean"].iloc[0]
    assert np.isclose(m, np.mean([6.0 + 0.1 * s for s in range(6)]))  # computed from the records, not invented
    st = pd.read_csv(tmp_path / "tables" / "E2_stats_proposed_vs_fedavg.csv")
    assert st["n"].iloc[0] == 6 and st["effect_r"].iloc[0] == -1.0  # proposed lower error in every seed
    deg = pd.read_csv(tmp_path / "tables" / "E8_degradation_vs_no_attack.csv")
    assert np.isclose(deg[deg["method"] == "fedavg"]["mean"].iloc[0], 3.0)
    w8 = pd.read_csv(tmp_path / "tables" / "E8_attacker_weights.csv")
    assert (w8[w8["method"] == "proposed"]["attacker_rel_weight"] < 0.5).all()
    fair = pd.read_csv(tmp_path / "tables" / "E6_fairness_weights.csv")
    assert set(fair["method"]) == {"fedavg", "proposed"}


def test_report_handles_missing_experiments(tmp_path) -> None:
    ec = load_eval_config("smoke")
    files = generate_report([_record("fedavg", 0, ["E1"], 6.0)], tmp_path, ec)
    assert (tmp_path / "summary.md").exists() and files
    assert "_no runs_" in (tmp_path / "summary.md").read_text()
    assert records_frame([]).empty and flatten_rounds([]).empty
    generate_report([], tmp_path / "empty", ec)
