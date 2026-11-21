"""Experiment cells (torch-free): the grid of E1-E8 and how a cell maps to a simulator configuration.

A cell is one (method, configuration, seed) run. Cells shared by several experiments are run once
(``Cell.id`` ignores the experiment tag) and the record lists every experiment it belongs to.

Methods: fedavg | proposed | proposed_reported | local_only | centralized | ekf | ukf | eskf.
``proposed`` = server-verified Layer 2 (+ clipping + floor); ``proposed_reported`` = trusts the client's own claims
(no server measurement, no clipping): the E8 baseline.

E7 (CARLA transfer) is NOT implemented: it needs CARLA, which cannot run here. The report states this explicitly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from afkn.eval.config import ScaleSpec
from afkn.sim.config import ClientSpec, SimConfig, SplitSpec

METHODS = ("fedavg", "proposed", "proposed_reported", "local_only", "centralized", "ekf", "ukf", "eskf")
FED_METHODS = ("fedavg", "proposed", "proposed_reported")
FILTER_METHODS = ("ekf", "ukf", "eskf")

#: confidence-spec overrides for ablation variants (E5) and for the "reported-signals-only" baseline (E8)
VARIANTS: dict[str, dict] = {
    "default": {},
    "no_direction": {"use_direction": False},
    "no_perf": {"use_perf": False},
    "no_claim": {"use_claim": False},
    "no_clip": {"use_clip": False},
    "no_history": {"use_history": False},
    "no_floor": {"weight_floor": 0.0},
    "combine_mean": {"combine": "mean"},
}
REPORTED_ONLY: dict = {"perf_source": "reported", "use_direction": False, "use_claim": False, "use_clip": False}


@dataclass(frozen=True)
class Cell:
    exp: str
    method: str
    seed: int
    noise: float = 1.0
    hetero: str = "strong"
    faulty_frac: float | None = None  # None = keep the default fleet; else this fraction of clients is faulty
    participation: float = 0.6
    attack: str = "none"
    attack_frac: float = 0.0
    variant: str = "default"

    @property
    def id(self) -> str:
        d = asdict(self)
        d.pop("exp")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def data_key(self) -> tuple:
        """Cells with equal data_key share one dataset (the dataset does not depend on method or attack)."""
        return (self.noise, self.hetero, self.faulty_frac, self.seed)


def _cell(exp: str, method: str, seed: int, **kw) -> Cell:
    if method not in METHODS:
        raise ValueError(f"unknown method {method}")
    return Cell(exp=exp, method=method, seed=seed, **kw)


def build_grid(sc: ScaleSpec, experiments: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5", "E6", "E8")) -> list[Cell]:
    """All cells for the requested experiments, de-duplicated by ``Cell.id`` (first experiment tag wins in the list;
    ``merge_exps`` below records every tag)."""
    out: list[Cell] = []
    for s in sc.seeds:
        if "E1" in experiments:
            for n in sc.noise_levels:
                for h in sc.hetero_levels:
                    out.append(_cell("E1", "fedavg", s, noise=n, hetero=h))
        if "E2" in experiments:
            for n in sc.noise_levels:
                for h in sc.hetero_levels:
                    for m in METHODS:
                        if m != "proposed_reported":
                            out.append(_cell("E2", m, s, noise=n, hetero=h))
        if "E3" in experiments:
            for ff in sc.faulty_fracs:
                for m in ("fedavg", "proposed"):
                    out.append(_cell("E3", m, s, faulty_frac=ff))
        if "E4" in experiments:
            for p in sc.participations:
                for m in ("fedavg", "proposed"):
                    out.append(_cell("E4", m, s, participation=p))
        if "E5" in experiments:
            for v in sc.ablation_variants:
                for a in sc.ablation_attacks:
                    frac = 0.0 if a == "none" else sc.ablation_attack_frac
                    out.append(_cell("E5", "proposed", s, attack=a, attack_frac=frac, variant=v))
        if "E6" in experiments:
            for m in ("fedavg", "proposed"):
                out.append(_cell("E6", m, s))
        if "E8" in experiments:
            for m in ("fedavg", "proposed_reported", "proposed"):
                out.append(_cell("E8", m, s))  # no-attack reference
                for a in sc.attacks:
                    for f in sc.attack_fracs:
                        out.append(_cell("E8", m, s, attack=a, attack_frac=f))
    return out


def merge_exps(cells: list[Cell]) -> dict[str, tuple[Cell, list[str]]]:
    """Group by ``Cell.id``: {id: (cell, sorted experiment tags)}."""
    out: dict[str, tuple[Cell, list[str]]] = {}
    for c in cells:
        if c.id in out:
            tags = out[c.id][1]
            if c.exp not in tags:
                tags.append(c.exp)
        else:
            out[c.id] = (c, [c.exp])
    return {k: (c, sorted(t)) for k, (c, t) in out.items()}


def build_sim_config(base: SimConfig, cell: Cell, sc: ScaleSpec) -> SimConfig:
    """Simulator configuration for a cell: scale sizes, noise level, heterogeneity and faulty fraction.

    hetero: ``homog`` = every client is mid_grade with identical routes; ``mild`` = high/mid/low cycle with the
    default routes; ``strong`` = the default five-profile fleet. ``noise`` multiplies each client's sigma_scale.
    ``faulty_frac`` replaces the fleet's faulty clients by mid_grade, then makes round(frac * n) clients faulty
    (constant_bias and random_spikes alternating, from the last client backwards).
    """
    n = len(base.clients)
    clients: list[ClientSpec] = []
    for i, c in enumerate(base.clients):
        if cell.hetero == "homog":
            c = replace(c, profile="mid_grade", route_mix=(0.5, 0.4, 0.1), speed_scale=1.0, turn_scale=1.0,
                        sigma_scale=1.0, fault_mode=None)
        elif cell.hetero == "mild":
            c = replace(c, profile=("high_grade", "mid_grade", "low_grade")[i % 3], fault_mode=None)
        elif cell.hetero != "strong":
            raise ValueError(f"unknown hetero level {cell.hetero}")
        clients.append(replace(c, sigma_scale=c.sigma_scale * cell.noise))
    if cell.faulty_frac is not None:
        clients = [replace(c, profile="mid_grade", fault_mode=None) if c.profile == "faulty" else c for c in clients]
        k = round(cell.faulty_frac * n)
        for j in range(k):
            idx = n - 1 - j
            mode = "constant_bias" if j % 2 == 0 else "random_spikes"
            clients[idx] = replace(clients[idx], profile="faulty", fault_mode=mode)
    sp = sc.splits
    return base.replace(
        steps=sc.steps,
        clients=tuple(clients),
        splits=SplitSpec(train=sp["train"], val=sp["val"], test=sp["test"], global_test=sp["global_test"], trusted=sp["trusted"]),
    )
