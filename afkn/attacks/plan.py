"""Model-update attacks by compromised vehicles.

A compromised vehicle trains honestly, then replaces its delta before signing (it holds a valid key,
so Layer 1 passes by design). Its self-reported signals are left as computed by the honest training
(it keeps claiming improvement), which is what the Layer 2 claim check has to catch.

    random      Gaussian noise, norm = random_gain x ||honest||
    sign_flip   -flip_k x honest
    scaling     scale x honest
    bias        honest + bias_gain x ||honest|| x u   (u: fixed unit direction per attacker)
    adaptive    stealthy: honest + beta ||honest|| u_perp (u_perp orthogonal to honest), rescaled to
                adaptive_scale x ||honest||. At the defaults it stays aligned with the honest/reference
                direction and has an honest-sized norm, so Layer 2 is expected to be partly blind to it.

ASSUMPTION: the plan's "bias" attack is implemented in parameter space (a constant systematic offset of
the update), not as GNSS data poisoning. The adaptive attacker approximates the server's reference
update by its own honest delta (it does not see the server's trusted set).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from afkn.attacks.config import AttackSpec
from afkn.fed.messages import ClientUpdate
from afkn.seeding import SeedTree


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def attack_delta(spec: AttackSpec, honest: np.ndarray, rng: np.random.Generator, bias_dir: np.ndarray) -> np.ndarray:
    """Pure function: the delta an attacker sends given its honest delta (float64 in, float64 out)."""
    h = np.asarray(honest, dtype=np.float64)
    hn = float(np.linalg.norm(h))
    a = spec.attack
    if a == "none":
        return h
    if a == "random":
        return _unit(rng.normal(size=h.shape)) * spec.random_gain * hn
    if a == "sign_flip":
        return -spec.flip_k * h
    if a == "scaling":
        return spec.scale * h
    if a == "bias":
        return h + spec.bias_gain * hn * bias_dir
    if a == "adaptive":
        u = bias_dir - (bias_dir @ _unit(h)) * _unit(h)  # remove the component along the honest direction
        out = h + spec.adaptive_beta * hn * _unit(u)
        return _unit(out) * spec.adaptive_scale * hn
    raise ValueError(f"unknown attack '{a}'")


class AttackPlan:
    """Which clients are compromised and how they alter their updates. Pure function of (spec, n, tree)."""

    def __init__(self, spec: AttackSpec, n_clients: int, tree: SeedTree) -> None:
        self.spec = spec
        self.tree = tree.child("attack")
        if spec.attack == "none":
            ids: tuple[int, ...] = ()
        elif spec.ids is not None:
            if max(spec.ids, default=-1) >= n_clients:
                raise ValueError("attacker id out of range")
            ids = tuple(sorted(spec.ids))
        else:
            k = max(1, round(spec.fraction * n_clients)) if spec.fraction > 0 else 0
            ids = tuple(sorted(int(i) for i in self.tree.child("ids").rng().permutation(n_clients)[:k]))
        self.attacker_ids = ids

    def is_attacker(self, client_id: int, round_idx: int) -> bool:
        return client_id in self.attacker_ids and round_idx >= self.spec.start_round

    def _bias_dir(self, client_id: int, size: int) -> np.ndarray:
        return _unit(self.tree.child("dir", client_id).rng().normal(size=size))

    def apply(self, upd: ClientUpdate, round_idx: int) -> ClientUpdate:
        """Return ``upd`` unchanged for honest clients / before ``start_round``, else the attacked update."""
        if not self.is_attacker(upd.client_id, round_idx):
            return upd
        honest = np.asarray(upd.delta, dtype=np.float64)
        rng = self.tree.child("noise", upd.client_id, round_idx).rng()
        out = attack_delta(self.spec, honest, rng, self._bias_dir(upd.client_id, honest.size))
        return dataclasses.replace(upd, delta=out.astype(np.float32))
