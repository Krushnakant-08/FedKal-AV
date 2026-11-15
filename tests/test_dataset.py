"""Federated dataset: spec sizes, determinism, disjointness, non-IID structure, mixed sets."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from afkn.data import build_client_data, build_federated_dataset
from afkn.scripts.inspect_dataset import main as inspect_main
from afkn.sim.batch import FIELD_NAMES
from afkn.sim.config import SimConfig, SplitSpec

PROFILES = ("high_grade", "mid_grade", "low_grade", "intermittent", "faulty")


@pytest.fixture(scope="module")
def small_ds(small_cfg: SimConfig):
    return build_federated_dataset(small_cfg)


def _equal(a, b) -> bool:
    return all(np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True) for f in FIELD_NAMES)


# ---------------------------------------------------------------------- structure


def test_sizes_and_contract_small(small_ds, small_cfg):
    assert len(small_ds.clients) == 10
    s = small_cfg.splits
    for c in small_ds.clients:
        assert (c.train.n, c.val.n, c.test.n) == (s.train, s.val, s.test)
        assert c.n_train == s.train
        for b in (c.train, c.val, c.test):
            assert b.T == small_cfg.steps and b.dt == small_cfg.dt
            assert np.array_equal(np.isnan(b.z).any(-1), ~b.z_mask)
            assert np.isfinite(b.x).all() and np.isfinite(b.u_imu).all()
    assert small_ds.global_test.n == s.global_test and small_ds.trusted.n == s.trusted


def test_client_profiles_cover_all_five_noise_classes(small_ds):
    assert {c.profile for c in small_ds.clients} == set(PROFILES)
    modes = {c.device.fault_mode for c in small_ds.clients if c.profile == "faulty"}
    assert modes == {"constant_bias", "random_spikes"}
    assert all(c.device.fault_mode is None for c in small_ds.clients if c.profile != "faulty")
    for c in small_ds.clients:  # profile_idx labels match the client's profile
        assert set(np.unique(c.train.profile_idx)) == {small_ds.cfg.profile_names.index(c.profile)}


# -------------------------------------------------------------------- determinism


def test_same_seed_bit_identical(small_cfg):
    a, b = build_federated_dataset(small_cfg), build_federated_dataset(small_cfg)
    assert a.fingerprint() == b.fingerprint()
    for (na, ba), (nb, bb) in zip(a.all_batches(), b.all_batches()):
        assert na == nb and ba.content_hash() == bb.content_hash() and _equal(ba, bb)


def test_different_seed_different_data(small_cfg, small_ds):
    other = build_federated_dataset(small_cfg, seed=small_cfg.seed + 1)
    assert other.fingerprint() != small_ds.fingerprint()
    for (_, ba), (_, bb) in zip(small_ds.all_batches(), other.all_batches()):
        assert ba.content_hash() != bb.content_hash()


def test_seed_argument_matches_config_seed(small_cfg):
    via_arg = build_federated_dataset(small_cfg, seed=5)
    via_cfg = build_federated_dataset(small_cfg.replace(seed=5))
    for (_, ba), (_, bb) in zip(via_arg.all_batches(), via_cfg.all_batches()):
        assert ba.content_hash() == bb.content_hash()


def test_each_client_is_independent_of_the_others(small_cfg, small_ds):
    for i in (0, 4, 9):
        solo = build_client_data(small_cfg, i)
        for split in ("train", "val", "test"):
            assert getattr(solo, split).content_hash() == getattr(small_ds.clients[i], split).content_hash()
        assert solo.device == small_ds.clients[i].device
    with pytest.raises(IndexError):
        build_client_data(small_cfg, 10)


# ------------------------------------------------------------------- disjointness


def test_all_trajectories_distinct_across_every_split_and_client(small_ds):
    seen: dict[str, str] = {}
    total = 0
    for name, b in small_ds.all_batches():
        for h in b.row_hashes():
            assert h not in seen, f"{name} shares a trajectory with {seen[h]}"
            seen[h] = name
            total += 1
    s = small_ds.cfg.splits
    assert total == 10 * (s.train + s.val + s.test) + s.global_test + s.trusted


def test_trusted_set_is_disjoint_from_clients_and_from_global_test(small_ds):
    trusted = set(small_ds.trusted.row_hashes())
    others = set(small_ds.global_test.row_hashes())
    for c in small_ds.clients:
        for b in (c.train, c.val, c.test):
            others |= set(b.row_hashes())
    assert not (trusted & others)


# -------------------------------------------------------------------- mixed sets


@pytest.mark.parametrize("which", ["global_test", "trusted"])
def test_mixed_sets_contain_every_profile_evenly(small_ds, which):
    b = getattr(small_ds, which)
    counts = [int((b.profile_idx == i).sum()) for i in range(len(PROFILES))]
    assert min(counts) >= b.n // len(PROFILES) and max(counts) - min(counts) <= 1
    assert sum(counts) == b.n
    # shuffled: profiles are interleaved, not stored in blocks
    assert (np.diff(b.profile_idx) != 0).sum() > len(PROFILES)


def test_uneven_mixed_set_size_is_handled(small_cfg):
    cfg = small_cfg.replace(splits=SplitSpec(train=2, val=1, test=1, global_test=7, trusted=3))
    ds = build_federated_dataset(cfg)
    assert ds.global_test.n == 7 and ds.trusted.n == 3
    assert sorted(np.bincount(ds.global_test.profile_idx, minlength=5)) == [1, 1, 1, 2, 2]


# ------------------------------------------------------------------------ non-IID


@pytest.fixture(scope="module")
def noniid_clients(default_cfg: SimConfig):
    cfg = default_cfg.replace(splits=SplitSpec(train=16, val=1, test=1, global_test=5, trusted=5))  # 1000 steps
    return cfg, [build_client_data(cfg, i) for i in range(len(cfg.clients))]


def test_routes_differ_across_clients_and_follow_each_clients_mix(noniid_clients):
    cfg, clients = noniid_clients
    fracs = np.array([[(c.train.seg == k).mean() for k in range(3)] for c in clients])
    want = np.array([cfg.clients[i].route_mix for i in range(len(clients))])
    want = want / want.sum(axis=1, keepdims=True)
    assert np.abs(fracs - want).max() < 0.15  # ~100 segments/client -> s.e. ~0.05
    assert fracs[:, 1].max() - fracs[:, 1].min() > 0.3  # highway share genuinely varies by client
    assert np.corrcoef(fracs[:, 2], want[:, 2])[0, 1] > 0.8


def test_speeds_differ_across_clients(noniid_clients):
    _, clients = noniid_clients
    v = np.array([c.train.x[..., 3].mean() for c in clients])
    assert clients[0].train.x[..., 3].mean() > 1.5 * clients[1].train.x[..., 3].mean()  # highway-heavy vs urban-heavy
    assert v.max() - v.min() > 5.0


def test_noise_levels_follow_profile_grade(noniid_clients):
    cfg, clients = noniid_clients
    sigma = {p: np.mean([c.device.gnss_sigma for c in clients if c.profile == p]) for p in PROFILES}
    assert sigma["high_grade"] < sigma["mid_grade"] < sigma["low_grade"]
    imu = {p: np.mean([c.device.imu_sigma[0] for c in clients if c.profile == p]) for p in PROFILES}
    assert imu["high_grade"] < imu["mid_grade"] < imu["low_grade"]
    # per-client noise scale makes same-profile clients differ too
    hi = [c.device.gnss_sigma for c in clients if c.profile == "high_grade"]
    assert hi[0] != hi[1]
    # intermittent clients really have less GNSS than high-grade ones
    avail = {p: np.mean([c.train.z_mask.mean() for c in clients if c.profile == p]) for p in PROFILES}
    assert avail["intermittent"] < avail["high_grade"] - 0.1


# ------------------------------------------------------------------ full spec size


@pytest.mark.slow
def test_full_spec_dataset_sizes_determinism_and_disjointness(default_cfg):
    ds = build_federated_dataset(default_cfg)
    assert len(ds.clients) == 10
    for c in ds.clients:
        assert (c.train.n, c.val.n, c.test.n) == (200, 40, 40)
        assert c.train.x.shape == (200, 1001, 4) and c.train.u_imu.shape == (200, 1000, 2)
        assert c.train.dt == 0.1
    assert ds.global_test.n == 200 and ds.trusted.n == 200
    hashes = [h for _, b in ds.all_batches() for h in b.row_hashes()]
    assert len(hashes) == 10 * 280 + 400 and len(set(hashes)) == len(hashes)
    # bit-identical regeneration of an arbitrary client at full size
    again = build_client_data(default_cfg, 7)
    assert again.train.content_hash() == ds.clients[7].train.content_hash()
    assert again.test.content_hash() == ds.clients[7].test.content_hash()


# --------------------------------------------------------------------- summary / CLI


def test_summary_is_json_serialisable_and_honest(small_ds):
    s = small_ds.summary()
    json.dumps(s)
    assert s["fingerprint"] == small_ds.fingerprint() and len(s["clients"]) == 10
    assert sum(s["trusted_profile_counts"].values()) == small_ds.trusted.n


def test_inspect_cli_runs_from_yaml_and_prints_fingerprint(small_cfg, tmp_path, capsys):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(small_cfg.to_dict()))))
    assert inspect_main(["--config", str(path), "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "seed=3" in out and "fingerprint=" in out and "faulty" in out
    assert inspect_main(["--config", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["seed"] == small_cfg.seed
