"""Round loop (FedAvg by default) and the local-only / centralized baselines it is compared with.

Everything is a pure function of (dataset, configs, seed). Messages go through their byte
encoding every round, so the byte counts logged are those of the serialized messages.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from afkn.data.partition import FederatedDataset
from afkn.attacks.plan import AttackPlan
from afkn.fed.aggregate import RoundContext, Weighting, fedavg_weights, normalize_weights, sample_participants
from afkn.fed.client import Client
from afkn.fed.config import FedSpec
from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.fed.server import Server
from afkn.models.config import KalmanNetConfig
from afkn.models.kalmannet import KalmanNet
from afkn.models.train import evaluate, train
from afkn.security.identities import Identities
from afkn.security.verifier import IdentityVerifier
from afkn.seeding import SeedTree
from afkn.sim.batch import TrajectoryBatch

#: weighting policy for the accepted updates of a round (default None = FedAvg's ``n_samples``);
#: may return raw scores or a ``Weighting`` (scores + per-client log rows + replacement deltas + floor)
Weighter = Callable[[list[ClientUpdate], RoundContext], "Weighting | np.ndarray"]


@dataclass
class RoundLog:
    """Everything logged for one federated round."""

    round: int
    participants: list[int]
    bytes_down: int  # P * 4 per participating client (project accounting rule)
    bytes_up: int  # P * 4 per participating client
    wire_down: int  # serialized message sizes incl. headers
    wire_up: int
    global_rmse: float  # NaN on rounds without evaluation
    clients: list[dict] = field(default_factory=list)  # per-client rows


@dataclass
class RunLog:
    mode: str  # "fedavg" | "local_only" | "centralized"
    curve: list[tuple[int, float]] = field(default_factory=list)  # (round, global-test position RMSE)
    rounds: list[RoundLog] = field(default_factory=list)  # federated mode only
    final_rmse: float = float("nan")
    per_client_final: list[float] = field(default_factory=list)  # local-only mode only
    params: np.ndarray | None = None  # final global parameters (fedavg / centralized)

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_down + r.bytes_up for r in self.rounds)


def _should_eval(r: int, fed: FedSpec) -> bool:
    return (r + 1) % fed.eval_every == 0 or r == fed.rounds - 1


def _init_params(dataset: FederatedDataset, kcfg: KalmanNetConfig, tree: SeedTree) -> np.ndarray:
    net = KalmanNet(kcfg, model=dataset.cfg.motion_model, dt=dataset.cfg.dt, seed=tree.child("init").int_seed())
    return net.get_flat_params()


def run_federated(
    dataset: FederatedDataset,
    kcfg: KalmanNetConfig,
    fed: FedSpec,
    *,
    seed: int,
    weighter: Weighter | None = None,
    init_params: np.ndarray | None = None,
    identities: Identities | None = None,
    attack: AttackPlan | None = None,
) -> RunLog:
    """FedAvg (weights ``n_i``) unless a ``weighter`` is given. Step 6 plugs the confidence scores in here."""
    tree = SeedTree(seed)
    cfg = dataset.cfg
    verifier = IdentityVerifier(identities.registry) if identities is not None else None
    server = Server(_init_params(dataset, kcfg, tree) if init_params is None else init_params, fed,
                    tree.child("server"), verifier)
    clients = [Client(c, kcfg, motion_model=cfg.motion_model, dt=cfg.dt) for c in dataset.clients]
    eval_net = KalmanNet(kcfg, model=cfg.motion_model, dt=cfg.dt, seed=0)
    log = RunLog(mode="fedavg")

    for r in range(fed.rounds):
        ids = sample_participants(len(clients), fed.participation, tree.child("sample", r).rng())
        bc = GlobalBroadcast.from_bytes(server.broadcast(r).to_bytes())  # through the wire encoding
        rows: list[dict] = []
        accepted: list[ClientUpdate] = []
        wire_up = 0
        for cid in ids:
            seed_c = tree.child("client", int(cid), "round", r).int_seed()
            upd = clients[int(cid)].local_update(bc, fed.local_epochs, seed_c)
            is_attacker = attack is not None and attack.is_attacker(int(cid), r)
            if attack is not None:
                upd = attack.apply(upd, r)  # compromised vehicle alters its delta, then signs with its valid key
            if identities is not None:
                upd = identities.sign(upd)
            raw_bytes = upd.to_bytes()
            wire_up += len(raw_bytes)
            upd = ClientUpdate.from_bytes(raw_bytes)
            reason = server.check_update(upd, bc)
            rows.append({
                "client": int(cid), "accepted": reason is None, "reject_reason": reason,
                "n_samples": upd.n_samples, "delta_norm": float(np.linalg.norm(upd.delta.astype(np.float64))),
                "bytes_up": upd.payload_bytes, "signals": dict(upd.signals), "weight": 0.0, "attacker": is_attacker,
            })
            if reason is None:
                accepted.append(upd)
        if accepted:
            agg_updates = accepted
            if weighter is None:
                w = server.final_weights(fedavg_weights([u.n_samples for u in accepted]))
                info: dict[int, dict] = {}
            else:
                wt = weighter(accepted, RoundContext(r, server.params.copy()))
                if not isinstance(wt, Weighting):
                    wt = Weighting(raw=np.asarray(wt, dtype=np.float64))
                w = normalize_weights(wt.raw, server.fed.weight_floor if wt.floor is None else wt.floor)
                info = wt.info
                if wt.deltas is not None:
                    agg_updates = [dataclasses.replace(u, delta=np.asarray(d, dtype=np.float32))
                                   for u, d in zip(accepted, wt.deltas, strict=True)]
            by_id = {u.client_id: float(wi) for u, wi in zip(accepted, w, strict=True)}
            for row in rows:
                row["weight"] = by_id.get(row["client"], 0.0)
                if row["client"] in info:
                    row["layer2"] = info[row["client"]]
            server.aggregate(agg_updates, w)
        rmse = float("nan")
        if _should_eval(r, fed):
            eval_net.set_flat_params(server.params)
            rmse = evaluate(eval_net, dataset.global_test)["position_rmse"]
            log.curve.append((r, rmse))
        k = len(ids)
        log.rounds.append(RoundLog(r, [int(i) for i in ids], k * bc.payload_bytes, k * server.params.size * 4,
                                   k * bc.wire_bytes, wire_up, rmse, rows))
    log.final_rmse = log.curve[-1][1]
    log.params = server.params.copy()
    return log


def run_local_only(dataset: FederatedDataset, kcfg: KalmanNetConfig, fed: FedSpec, *, seed: int,
                   init_params: np.ndarray | None = None) -> RunLog:
    """Every client trains alone on its own train split (all clients, every round, same epochs per round).
    ``curve`` holds the mean global-test RMSE over clients."""
    tree = SeedTree(seed)
    cfg = dataset.cfg
    p0 = _init_params(dataset, kcfg, tree) if init_params is None else init_params
    nets = []
    for _ in dataset.clients:
        n = KalmanNet(kcfg, model=cfg.motion_model, dt=cfg.dt, seed=0)
        n.set_flat_params(p0)
        nets.append(n)
    tcfg = dataclasses.replace(kcfg.train, epochs=fed.local_epochs)
    log = RunLog(mode="local_only")
    for r in range(fed.rounds):
        for c, net in zip(dataset.clients, nets, strict=True):
            train(net, c.train, tcfg, seed=tree.child("client", c.client_id, "round", r).int_seed())
        if _should_eval(r, fed):
            vals = [evaluate(n, dataset.global_test)["position_rmse"] for n in nets]
            log.curve.append((r, float(np.mean(vals))))
            log.per_client_final = vals
    log.final_rmse = log.curve[-1][1]
    return log


def run_centralized(dataset: FederatedDataset, kcfg: KalmanNetConfig, fed: FedSpec, *, seed: int,
                    init_params: np.ndarray | None = None) -> RunLog:
    """Reference upper bound: all clients' train data pooled (privacy-violating by design), same epochs per round."""
    tree = SeedTree(seed)
    cfg = dataset.cfg
    net = KalmanNet(kcfg, model=cfg.motion_model, dt=cfg.dt, seed=0)
    net.set_flat_params(_init_params(dataset, kcfg, tree) if init_params is None else init_params)
    pooled = TrajectoryBatch.concat([c.train for c in dataset.clients])
    tcfg = dataclasses.replace(kcfg.train, epochs=fed.local_epochs)
    log = RunLog(mode="centralized")
    for r in range(fed.rounds):
        train(net, pooled, tcfg, seed=tree.child("central", "round", r).int_seed())
        if _should_eval(r, fed):
            log.curve.append((r, evaluate(net, dataset.global_test)["position_rmse"]))
    log.final_rmse = log.curve[-1][1]
    log.params = net.get_flat_params()
    return log


def run_comparison(dataset: FederatedDataset, kcfg: KalmanNetConfig, fed: FedSpec, *, seed: int) -> dict:
    """Federated vs local-only vs centralized from one shared initialisation; the gap is logged, not asserted."""
    p0 = _init_params(dataset, kcfg, SeedTree(seed))
    runs = {
        "fedavg": run_federated(dataset, kcfg, fed, seed=seed, init_params=p0),
        "local_only": run_local_only(dataset, kcfg, fed, seed=seed, init_params=p0),
        "centralized": run_centralized(dataset, kcfg, fed, seed=seed, init_params=p0),
    }
    f, lo, ce = (runs[k].final_rmse for k in ("fedavg", "local_only", "centralized"))
    summary = {
        "fedavg_rmse": f, "local_only_mean_rmse": lo, "centralized_rmse": ce,
        "gap_fedavg_minus_centralized": f - ce,
        "fedavg_beats_local_only": bool(f < lo),
        "total_bytes_fedavg": runs["fedavg"].total_bytes,
    }
    return {"runs": runs, "summary": summary}
