"""Layer 1 tests (torch-free). Forged, tampered, replayed, unenrolled, revoked, stale, duplicate and
malformed updates must be rejected 100% of the time; a genuine update must be accepted."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.security import (
    REASONS,
    Identities,
    IdentityVerifier,
    KeyRegistry,
    RegistryError,
    generate_private_key,
    public_bytes,
    sign_update,
    verify_signature,
)
from afkn.seeding import SeedTree

P = 12


def _bc(rnd: int = 0, fill: float = 0.5) -> GlobalBroadcast:
    return GlobalBroadcast(round=rnd, nonce=bytes([rnd + 1]) * 16, params=np.full(P, fill, np.float32))


def _upd(bc: GlobalBroadcast, cid: int = 2, **kw) -> ClientUpdate:
    base = {"client_id": cid, "round": bc.round, "nonce": bc.nonce, "model_hash": bc.model_hash,
            "delta": np.linspace(-1, 1, P).astype(np.float32), "n_samples": 50,
            "signals": {"val_rmse_before": 4.0, "val_rmse_after": 3.0}}
    base.update(kw)
    return ClientUpdate(**base)


@pytest.fixture()
def fleet():
    ids = Identities(5, SeedTree(0))
    ver = IdentityVerifier(ids.registry)
    bc = _bc(0)
    ver.begin_round(bc)
    return ids, ver, bc


def test_genuine_update_accepted_through_the_wire(fleet) -> None:
    ids, ver, bc = fleet
    wire = ids.sign(_upd(bc)).to_bytes()
    res = ver.check(ClientUpdate.from_bytes(wire))
    assert res.ok and res.reason is None


def test_forged_id_rejected(fleet) -> None:
    ids, ver, bc = fleet
    # attacker holds key 4 but claims to be vehicle 2
    forged = sign_update(_upd(bc, cid=2), ids.keys[4])
    assert ver.check(forged).reason == "bad_signature"
    assert ver.check(_upd(bc, cid=2)).reason == "bad_signature"  # unsigned
    assert ver.check(_upd(bc).with_signature(b"\x00" * 64)).reason == "bad_signature"
    assert ver.check(_upd(bc).with_signature(b"short")).reason == "bad_signature"


def test_forger_cannot_lock_out_the_real_vehicle(fleet) -> None:
    ids, ver, bc = fleet
    assert ver.check(sign_update(_upd(bc, cid=2), ids.keys[4])).reason == "bad_signature"
    assert ver.check(ids.sign(_upd(bc, cid=2))).ok  # the genuine one still gets through


def test_tampering_with_any_field_rejected_100_percent(fleet) -> None:
    ids, ver, bc = fleet
    good = ids.sign(_upd(bc))
    d2 = good.delta.copy()
    d2[3] += 1e-3
    tampered = [
        ClientUpdate(good.client_id, good.round, good.nonce, good.model_hash, d2, good.n_samples, good.signals, good.signature),
        ClientUpdate(good.client_id, good.round, good.nonce, good.model_hash, good.delta, good.n_samples + 1, good.signals, good.signature),
        ClientUpdate(good.client_id, good.round, good.nonce, good.model_hash, good.delta, good.n_samples, {"val_rmse_before": 4.0, "val_rmse_after": 0.1}, good.signature),
        ClientUpdate(good.client_id, good.round + 1, good.nonce, good.model_hash, good.delta, good.n_samples, good.signals, good.signature),
        ClientUpdate(good.client_id, good.round, bytes(16), good.model_hash, good.delta, good.n_samples, good.signals, good.signature),
        ClientUpdate(good.client_id, good.round, good.nonce, bytes(32), good.delta, good.n_samples, good.signals, good.signature),
    ]
    assert all(ver.check(t).reason == "bad_signature" for t in tampered)
    rng = np.random.default_rng(0)  # 200 random single-bit flips of the serialized message
    raw = bytearray(good.to_bytes())
    rejected = 0
    for _ in range(200):
        b = bytearray(raw)
        b[int(rng.integers(len(b)))] ^= 1 << int(rng.integers(8))
        try:
            msg = ClientUpdate.from_bytes(bytes(b))
        except Exception:
            rejected += 1  # malformed on the wire: never reaches the verifier
            continue
        rejected += int(not ver.check(msg).ok)
    assert rejected == 200


def test_unenrolled_rejected(fleet) -> None:
    ids, ver, bc = fleet
    stranger = generate_private_key(SeedTree(99))
    assert ver.check(sign_update(_upd(bc, cid=77), stranger)).reason == "unenrolled"  # valid sig, unknown id
    assert ver.check(sign_update(_upd(bc, cid=2), stranger)).reason == "bad_signature"  # known id, unknown key


def test_revoked_rejected(fleet) -> None:
    ids, ver, bc = fleet
    ids.registry.revoke(2)
    assert ver.check(ids.sign(_upd(bc, cid=2))).reason == "revoked"
    assert ver.check(ids.sign(_upd(bc, cid=3))).ok


def test_replay_rejected(fleet) -> None:
    ids, ver, bc0 = fleet
    old = ids.sign(_upd(bc0))
    assert ver.check(old).ok
    bc1 = _bc(1)
    ver.begin_round(bc1)
    assert ver.check(old).reason == "replay_round"  # verbatim replay of last round's message
    # attacker with a valid key re-signs old content for the new round number but keeps the old nonce
    resigned = ids.sign(_upd(bc1, nonce=bc0.nonce))
    assert ver.check(resigned).reason == "replay_nonce"
    # same nonce but wrong round number
    assert ver.check(ids.sign(_upd(bc0, cid=3))).reason == "replay_round"


def test_stale_or_fake_model_rejected(fleet) -> None:
    ids, ver, bc = fleet
    other = _bc(0, fill=0.75)
    assert ver.check(ids.sign(_upd(bc, model_hash=other.model_hash))).reason == "stale_model"


def test_duplicate_in_round_rejected_then_allowed_next_round(fleet) -> None:
    ids, ver, bc = fleet
    assert ver.check(ids.sign(_upd(bc))).ok
    assert ver.check(ids.sign(_upd(bc))).reason == "duplicate"
    assert ver.check(ids.sign(_upd(bc, delta=np.zeros(P, np.float32)))).reason == "duplicate"
    bc1 = _bc(1)
    ver.begin_round(bc1)
    assert ver.check(ids.sign(_upd(bc1))).ok


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_delta_rejected(fleet, bad: float) -> None:
    ids, ver, bc = fleet
    d = np.zeros(P, np.float32)
    d[5] = bad
    assert ver.check(ids.sign(_upd(bc, delta=d))).reason == "non_finite"


def test_wrong_shape_rejected(fleet) -> None:
    ids, ver, bc = fleet
    assert ver.check(ids.sign(_upd(bc, delta=np.zeros(P + 1, np.float32)))).reason == "bad_shape"
    assert ver.check(ids.sign(_upd(bc, cid=3, delta=np.zeros(0, np.float32)))).reason == "bad_shape"


def test_rejected_update_does_not_mark_sender_as_seen(fleet) -> None:
    ids, ver, bc = fleet
    d = np.zeros(P, np.float32)
    d[0] = np.nan
    assert ver.check(ids.sign(_upd(bc, delta=d))).reason == "non_finite"
    assert ver.check(ids.sign(_upd(bc))).ok


def test_all_reasons_are_reachable_names() -> None:
    assert len(set(REASONS)) == len(REASONS) == 9


def test_check_requires_begin_round() -> None:
    ids = Identities(2, SeedTree(0))
    with pytest.raises(RuntimeError):
        IdentityVerifier(ids.registry).check(_upd(_bc()))


def test_registry_rules() -> None:
    reg = KeyRegistry()
    k = public_bytes(generate_private_key())
    reg.enroll(1, k)
    assert reg.is_enrolled(1) and not reg.is_revoked(1) and reg.public_key(1) == k and len(reg) == 1
    with pytest.raises(RegistryError):
        reg.enroll(1, public_bytes(generate_private_key()))  # no silent key replacement
    with pytest.raises(RegistryError):
        reg.enroll(2, b"short")
    with pytest.raises(RegistryError):
        reg.revoke(9)
    reg.revoke(1)
    assert reg.is_revoked(1)


def test_keys_are_deterministic_from_seed_and_distinct() -> None:
    a, b = Identities(3, SeedTree(5)), Identities(3, SeedTree(5))
    assert [public_bytes(k) for k in a.keys.values()] == [public_bytes(k) for k in b.keys.values()]
    assert len({public_bytes(k) for k in a.keys.values()}) == 3
    assert public_bytes(Identities(1, SeedTree(6)).keys[0]) != public_bytes(a.keys[0])


def test_verify_signature_direct() -> None:
    key = generate_private_key()
    bc = _bc()
    u = sign_update(_upd(bc), key)
    assert verify_signature(u, public_bytes(key))
    assert not verify_signature(u, public_bytes(generate_private_key()))
    assert not verify_signature(u, b"x")
