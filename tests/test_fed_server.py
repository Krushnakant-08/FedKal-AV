"""Server broadcast / structural checks / aggregation (torch-free)."""

from __future__ import annotations

import numpy as np

from afkn.fed import FedSpec
from afkn.fed.messages import ClientUpdate
from afkn.fed.server import Server
from afkn.seeding import SeedTree


def _server(p: int = 6, **fed) -> Server:
    return Server(np.arange(p, dtype=np.float32), FedSpec(**fed), SeedTree(0))


def _upd(bc, **kw) -> ClientUpdate:
    base = dict(client_id=1, round=bc.round, nonce=bc.nonce, model_hash=bc.model_hash,
                delta=np.ones(bc.params.size, np.float32), n_samples=10, signals={})
    base.update(kw)
    return ClientUpdate(**base)


def test_broadcast_deterministic_and_fresh_per_round() -> None:
    s1, s2 = _server(), _server()
    assert s1.broadcast(0).nonce == s2.broadcast(0).nonce
    assert s1.broadcast(0).nonce != s1.broadcast(1).nonce
    assert s1.broadcast(2).round == 2


def test_broadcast_is_a_copy() -> None:
    s = _server()
    bc = s.broadcast(0)
    s.params[0] = 99.0
    assert bc.params[0] == 0.0


def test_basic_checks_report_reasons() -> None:
    s = _server()
    bc = s.broadcast(5)
    assert s.basic_checks(_upd(bc), bc) is None
    assert s.basic_checks(_upd(bc, round=4), bc) == "round_mismatch"
    assert s.basic_checks(_upd(bc, nonce=bytes(16)), bc) == "nonce_mismatch"
    assert s.basic_checks(_upd(bc, model_hash=bytes(32)), bc) == "model_hash_mismatch"
    assert s.basic_checks(_upd(bc, delta=np.ones(3, np.float32)), bc) == "bad_shape"
    d = np.ones(6, np.float32)
    d[1] = np.inf
    assert s.basic_checks(_upd(bc, delta=d), bc) == "non_finite"


def test_aggregate_weighted_and_empty() -> None:
    s = _server(server_lr=1.0)
    bc = s.broadcast(0)
    ups = [_upd(bc, delta=np.full(6, 1.0, np.float32)), _upd(bc, client_id=2, delta=np.full(6, 3.0, np.float32))]
    before = s.params.copy()
    s.aggregate([], np.array([]))
    np.testing.assert_array_equal(s.params, before)  # nothing accepted: unchanged
    s.aggregate(ups, np.array([0.75, 0.25]))
    np.testing.assert_allclose(s.params, before + 1.5)


def test_final_weights_enforce_floor() -> None:
    s = _server(weight_floor=0.05)
    w = s.final_weights(np.array([0.0, 0.0, 10.0]))
    assert np.isclose(w.sum(), 1.0) and w.min() >= 0.05 - 1e-12
