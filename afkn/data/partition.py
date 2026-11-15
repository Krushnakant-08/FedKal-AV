"""Non-IID federated partition.

Seed namespaces (all derived from one root :class:`~afkn.seeding.SeedTree`)::

    ("client", i, "device")            per-client sensor constants
    ("client", i, "split", name)       train / val / test trajectories of client i
    ("global_test", ...)               shared test set, mixes all profiles
    ("trusted", ...)                   server-only set, mixes all profiles

Distinct top-level namespaces mean the global-test and trusted sets can never
coincide with each other or with any client split. ``tests/test_dataset.py``
also checks this empirically by hashing every trajectory.

Client ``i`` can be built alone (``build_client_data``) and is bit-identical to
the same client inside the full dataset: no client depends on any other.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from afkn.seeding import SeedTree
from afkn.sim.batch import TrajectoryBatch
from afkn.sim.config import ClientSpec, SimConfig
from afkn.sim.generator import generate_batch
from afkn.sim.sensors import DeviceParams, realize_device

SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class ClientData:
    """One client's non-IID data: realised device, spec and its three splits."""

    client_id: int
    spec: ClientSpec
    device: DeviceParams
    train: TrajectoryBatch
    val: TrajectoryBatch
    test: TrajectoryBatch

    @property
    def profile(self) -> str:
        return self.spec.profile

    @property
    def n_train(self) -> int:
        """Number of training trajectories (the ``n_samples`` a client reports)."""
        return self.train.n


@dataclass(frozen=True)
class FederatedDataset:
    cfg: SimConfig
    seed: int
    clients: tuple[ClientData, ...]
    global_test: TrajectoryBatch
    trusted: TrajectoryBatch

    def all_batches(self) -> list[tuple[str, TrajectoryBatch]]:
        out: list[tuple[str, TrajectoryBatch]] = []
        for c in self.clients:
            out += [(f"client{c.client_id}.{s}", getattr(c, s)) for s in SPLIT_NAMES]
        out += [("global_test", self.global_test), ("trusted", self.trusted)]
        return out

    def fingerprint(self) -> str:
        """SHA-256 over the config hash, the seed and every batch's content hash."""
        h = hashlib.sha256()
        h.update(f"cfg={self.cfg.config_hash()}|seed={self.seed}".encode())
        for name, b in self.all_batches():
            h.update(f"{name}={b.content_hash()}".encode())
        return h.hexdigest()

    def summary(self) -> dict:
        """Counts and realised statistics (all computed from the generated data)."""
        rows = []
        for c in self.clients:
            b = c.train
            rows.append(
                {
                    "client": c.client_id,
                    "profile": c.profile,
                    "fault_mode": c.device.fault_mode,
                    "n_train": c.train.n,
                    "n_val": c.val.n,
                    "n_test": c.test.n,
                    "gnss_sigma": round(c.device.gnss_sigma, 4),
                    "gnss_available": round(float(b.z_mask.mean()), 4),
                    "seg_frac_urban_highway_tunnel": [round(float((b.seg == k).mean()), 3) for k in range(3)],
                    "mean_speed": round(float(b.x[..., 3].mean()), 3),
                }
            )
        names = self.cfg.profile_names
        return {
            "seed": self.seed,
            "config_hash": self.cfg.config_hash(),
            "fingerprint": self.fingerprint(),
            "steps": self.cfg.steps,
            "dt": self.cfg.dt,
            "motion_model": self.cfg.motion_model,
            "global_test_n": self.global_test.n,
            "trusted_n": self.trusted.n,
            "global_test_profile_counts": {n: int((self.global_test.profile_idx == i).sum()) for i, n in enumerate(names)},
            "trusted_profile_counts": {n: int((self.trusted.profile_idx == i).sum()) for i, n in enumerate(names)},
            "clients": rows,
        }


def build_client_data(cfg: SimConfig, client_id: int, seed: int | None = None) -> ClientData:
    """Build one client's data. Independent of every other client."""
    if not 0 <= client_id < len(cfg.clients):
        raise IndexError(f"client_id {client_id} out of range for {len(cfg.clients)} clients")
    root = SeedTree(cfg.seed if seed is None else seed)
    spec = cfg.clients[client_id]
    ctree = root.child("client", client_id)
    device = realize_device(
        spec.profile,
        cfg.profiles[spec.profile],
        ctree.child("device").rng(),
        sigma_scale=spec.sigma_scale,
        fault_mode=spec.fault_mode,
    )
    pidx = cfg.profile_names.index(spec.profile)
    sizes = {"train": cfg.splits.train, "val": cfg.splits.val, "test": cfg.splits.test}
    batches = {
        name: generate_batch(
            cfg,
            device,
            spec.route_mix,
            sizes[name],
            ctree.child("split", name),
            speed_scale=spec.speed_scale,
            turn_scale=spec.turn_scale,
            profile_idx=pidx,
        )
        for name in SPLIT_NAMES
    }
    return ClientData(client_id, spec, device, **batches)


def build_mixed_set(cfg: SimConfig, tree: SeedTree, n: int) -> TrajectoryBatch:
    """A set mixing all profiles (as evenly as possible), built from fresh devices and shuffled.

    Each profile uses its profile-default fault mode (``faulty`` -> its config default).
    """
    names = cfg.profile_names
    base, extra = divmod(n, len(names))
    parts: list[TrajectoryBatch] = []
    for i, name in enumerate(names):
        count = base + (1 if i < extra else 0)
        if count == 0:
            continue
        dev = realize_device(name, cfg.profiles[name], tree.child("device", name).rng())
        parts.append(
            generate_batch(cfg, dev, cfg.mixed_route_mix, count, tree.child("batch", name), profile_idx=i)
        )
    if not parts:
        raise ValueError("cannot build an empty mixed set")
    merged = TrajectoryBatch.concat(parts)
    return merged.subset(tree.child("shuffle").rng().permutation(merged.n))


def build_federated_dataset(cfg: SimConfig, seed: int | None = None) -> FederatedDataset:
    """Build the full dataset: all clients, the shared global test set, and the server trusted set."""
    s = cfg.seed if seed is None else seed
    root = SeedTree(s)
    clients = tuple(build_client_data(cfg, i, s) for i in range(len(cfg.clients)))
    global_test = build_mixed_set(cfg, root.child("global_test"), cfg.splits.global_test)
    trusted = build_mixed_set(cfg, root.child("trusted"), cfg.splits.trusted)
    return FederatedDataset(cfg=cfg, seed=s, clients=clients, global_test=global_test, trusted=trusted)
