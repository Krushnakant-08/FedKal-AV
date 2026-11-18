"""Run experiment cells and write raw JSON records (needs PyTorch).

One JSON file per cell: ``<out>/cells/<cell id>.json``. Existing files are kept (resume). No number in a record
comes from anywhere but the run itself.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from afkn.attacks import AttackPlan, AttackSpec
from afkn.confidence import ConfidenceAggregator, ConfidenceSpec
from afkn.confidence.oracle_torch import TorchOracle
from afkn.data.partition import FederatedDataset, build_federated_dataset
from afkn.eval.cells import (
    FILTER_METHODS,
    REPORTED_ONLY,
    VARIANTS,
    Cell,
    build_sim_config,
    merge_exps,
)
from afkn.eval.config import EvalConfig
from afkn.eval.metrics import pooled_rmse, rmse_by_profile
from afkn.fed.config import FedSpec
from afkn.fed.loop import run_centralized, run_federated, run_local_only
from afkn.filters import EKF, ESKF, UKF, matched_params, matched_params_by_profile
from afkn.models.config import KalmanNetConfig, load_kalmannet_config
from afkn.models.kalmannet import KalmanNet
from afkn.security.identities import Identities
from afkn.seeding import SeedTree
from afkn.sim.config import SimConfig, load_config


def kalmannet_config(ec: EvalConfig) -> KalmanNetConfig:
    k = ec.scale.kalmannet
    return load_kalmannet_config(overrides={
        "net": {"init_pos_gain": k["init_pos_gain"]},
        "train": {"lr": k["lr"], "batch_size": k["batch_size"], "chunk_len": k["chunk_len"]},
    })


def confidence_spec(cell: Cell) -> ConfidenceSpec:
    over = dict(REPORTED_ONLY) if cell.method == "proposed_reported" else dict(VARIANTS[cell.variant])
    return ConfidenceSpec(**over)


def _eval_model(params: np.ndarray, kcfg: KalmanNetConfig, cfg: SimConfig, batches: dict) -> dict:
    net = KalmanNet(kcfg, model=cfg.motion_model, dt=cfg.dt, seed=0)
    net.set_flat_params(params)
    return {k: net.run(b).x_hat for k, b in batches.items()}


def _record_common(ds: FederatedDataset, cfg: SimConfig) -> dict:
    return {"profiles": [c.profile for c in ds.clients], "profile_names": list(cfg.profile_names)}


def run_cell(cell: Cell, ec: EvalConfig, base_cfg: SimConfig, cache: dict) -> dict:
    """Execute one cell and return its JSON-serialisable record."""
    t0 = time.perf_counter()
    sc = ec.scale
    cfg = build_sim_config(base_cfg, cell, sc)
    dkey = cell.data_key()
    if dkey not in cache:
        cache.clear()  # keep one dataset in memory (cells are run grouped by data_key)
        cache[dkey] = build_federated_dataset(cfg, cell.seed)
    ds: FederatedDataset = cache[dkey]
    kcfg = kalmannet_config(ec)
    names = list(cfg.profile_names)
    rec: dict = {"id": cell.id, "cell": cell.__dict__.copy(), "method": cell.method, **_record_common(ds, cfg),
                 "curve": [], "total_bytes": 0, "rounds": [], "attackers": []}

    if cell.method in FILTER_METHODS:
        cls = {"ekf": EKF, "ukf": UKF, "eskf": ESKF}[cell.method]
        res = cls(matched_params_by_profile(cfg, ds.global_test.profile_idx)).run(ds.global_test)
        rec["final_rmse"] = pooled_rmse(res.x_hat, ds.global_test)
        rec["per_profile_rmse"] = rmse_by_profile(res.x_hat, ds.global_test, names)
        rec["client_test_rmse"] = [
            pooled_rmse(cls(matched_params(cfg, c.device)).run(c.test).x_hat, c.test) for c in ds.clients
        ]
    else:
        fed = FedSpec(rounds=sc.rounds, local_epochs=sc.local_epochs, participation=cell.participation,
                      eval_every=sc.eval_every, weight_floor=0.0)
        if cell.method == "local_only":
            log = run_local_only(ds, kcfg, fed, seed=cell.seed)
            params_list = log.client_params
            rec["final_rmse"] = log.final_rmse
            rec["client_test_rmse"] = [
                pooled_rmse(_eval_model(p, kcfg, cfg, {"t": c.test})["t"], c.test)
                for p, c in zip(params_list, ds.clients, strict=True)
            ]
            per_prof = [rmse_by_profile(_eval_model(p, kcfg, cfg, {"g": ds.global_test})["g"], ds.global_test, names)
                        for p in params_list]
            rec["per_profile_rmse"] = {n: float(np.mean([d[n] for d in per_prof if n in d])) for n in per_prof[0]}
            rec["curve"] = [[int(r), float(v)] for r, v in log.curve]
        else:
            if cell.method == "centralized":
                log = run_centralized(ds, kcfg, fed, seed=cell.seed)
            else:
                weighter = None
                if cell.method in ("proposed", "proposed_reported"):
                    spec = confidence_spec(cell)
                    oracle = TorchOracle(kcfg, ds.trusted, motion_model=cfg.motion_model, dt=cfg.dt,
                                         ref_epochs=spec.ref_epochs, seed=cell.seed)
                    weighter = ConfidenceAggregator(spec, oracle)
                plan = None
                if cell.attack != "none":
                    aspec = AttackSpec(attack=cell.attack, fraction=cell.attack_frac, **ec.attack_strengths[cell.attack])
                    plan = AttackPlan(aspec, len(ds.clients), SeedTree(cell.seed))
                    rec["attackers"] = list(plan.attacker_ids)
                ids = Identities(len(ds.clients), SeedTree(cell.seed).child("identities"))
                log = run_federated(ds, kcfg, fed, seed=cell.seed, weighter=weighter, identities=ids, attack=plan)
                rec["total_bytes"] = log.total_bytes
                rec["rounds"] = [_round_record(rl, ds) for rl in log.rounds]
            rec["final_rmse"] = log.final_rmse
            rec["curve"] = [[int(r), float(v)] for r, v in log.curve]
            xs = _eval_model(log.params, kcfg, cfg, {"g": ds.global_test, **{f"c{c.client_id}": c.test for c in ds.clients}})
            rec["per_profile_rmse"] = rmse_by_profile(xs["g"], ds.global_test, names)
            rec["client_test_rmse"] = [pooled_rmse(xs[f"c{c.client_id}"], c.test) for c in ds.clients]
    rec["elapsed_s"] = time.perf_counter() - t0
    return rec


def _round_record(rl, ds: FederatedDataset) -> dict:
    rows = []
    for r in rl.clients:
        row = {k: r[k] for k in ("client", "accepted", "reject_reason", "weight", "attacker", "n_samples",
                                 "delta_norm", "bytes_up", "signals")}
        row["profile"] = ds.clients[r["client"]].profile
        if "layer2" in r:
            row["layer2"] = r["layer2"]
        rows.append(row)
    return {"round": rl.round, "participants": rl.participants, "bytes_down": rl.bytes_down, "bytes_up": rl.bytes_up,
            "global_rmse": None if not np.isfinite(rl.global_rmse) else rl.global_rmse, "clients": rows}


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def run_cells(cells: list[Cell], ec: EvalConfig, out_dir: str | Path, *, base_cfg: SimConfig | None = None,
              log=print) -> list[Path]:
    """Run every unique cell (resuming from existing files). Returns the cell file paths in grid order."""
    out = Path(out_dir) / "cells"
    out.mkdir(parents=True, exist_ok=True)
    base = base_cfg or load_config()
    merged = merge_exps(cells)
    cache: dict = {}
    paths: list[Path] = []
    ordered = sorted(merged.items(), key=lambda kv: (str(kv[1][0].data_key()), kv[0]))
    for i, (cid, (cell, exps)) in enumerate(ordered):
        path = out / f"{cid}.json"
        paths.append(path)
        if path.exists():
            continue
        log(f"[{i + 1}/{len(ordered)}] {cell.method} seed={cell.seed} {exps} {cell.__dict__}")
        rec = run_cell(cell, ec, base, cache)
        rec["exps"] = exps
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, default=_json_default), encoding="utf-8")
        tmp.replace(path)
    return paths
