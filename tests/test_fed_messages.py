"""Byte-level messages (torch-free): round trips, strict parsing, signing payload coverage."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.fed.messages import ClientUpdate, GlobalBroadcast, MessageError, hash_params

NONCE = bytes(range(16))


def _bc(p: int = 10) -> GlobalBroadcast:
    return GlobalBroadcast(round=3, nonce=NONCE, params=np.arange(p, dtype=np.float32) / 7)


def _upd(p: int = 10, **kw) -> ClientUpdate:
    bc = _bc(p)
    base = dict(client_id=4, round=3, nonce=NONCE, model_hash=bc.model_hash,
                delta=np.linspace(-1, 1, p).astype(np.float32), n_samples=200,
                signals={"val_rmse_before": 5.5, "val_rmse_after": 4.25, "innov_ms": 2.0})
    base.update(kw)
    return ClientUpdate(**base)


def test_broadcast_roundtrip_and_hash() -> None:
    bc = _bc()
    out = GlobalBroadcast.from_bytes(bc.to_bytes())
    assert out.round == 3 and out.nonce == NONCE
    np.testing.assert_array_equal(out.params, bc.params)
    assert out.model_hash == hash_params(bc.params)
    assert bc.payload_bytes == 40 and bc.wire_bytes == len(bc.to_bytes()) > 40


def test_update_roundtrip_bit_exact_with_nan_signal() -> None:
    u = _upd(signals={"a": float("nan"), "b": -0.0}).with_signature(b"sig" * 10)
    out = ClientUpdate.from_bytes(u.to_bytes())
    assert out.to_bytes() == u.to_bytes()
    assert out.client_id == 4 and out.round == 3 and out.n_samples == 200 and out.signature == b"sig" * 10
    assert np.isnan(out.signals["a"]) and out.signals["b"] == 0.0
    np.testing.assert_array_equal(out.delta, u.delta)
    assert u.payload_bytes == 40


def test_serialization_independent_of_signal_insertion_order() -> None:
    a = _upd(signals={"x": 1.0, "y": 2.0})
    b = _upd(signals={"y": 2.0, "x": 1.0})
    assert a.to_bytes() == b.to_bytes() and a.signing_payload() == b.signing_payload()


@pytest.mark.parametrize("cut", [0, 3, 20, 60, -1])
def test_truncated_rejected(cut: int) -> None:
    raw = _upd().to_bytes()
    with pytest.raises(MessageError):
        ClientUpdate.from_bytes(raw[:cut])
    raw_b = _bc().to_bytes()
    with pytest.raises(MessageError):
        GlobalBroadcast.from_bytes(raw_b[:cut])


def test_trailing_bytes_and_bad_magic_rejected() -> None:
    with pytest.raises(MessageError):
        ClientUpdate.from_bytes(_upd().to_bytes() + b"\x00")
    with pytest.raises(MessageError):
        ClientUpdate.from_bytes(b"XXXX" + _upd().to_bytes()[4:])
    with pytest.raises(MessageError):
        GlobalBroadcast.from_bytes(_bc().to_bytes() + b"\x00")


def test_broadcast_hash_mismatch_rejected() -> None:
    raw = bytearray(_bc().to_bytes())
    raw[-1] ^= 0x01  # flip a parameter bit, hash no longer matches
    with pytest.raises(MessageError):
        GlobalBroadcast.from_bytes(bytes(raw))


def test_signing_payload_covers_every_field_and_ignores_signature() -> None:
    base = _upd()
    p = base.signing_payload()
    d2 = base.delta.copy()
    d2[0] += 1e-3
    variants = [
        _upd(client_id=5), _upd(round=4), _upd(nonce=bytes(16)), _upd(model_hash=bytes(32)),
        _upd(n_samples=201), _upd(signals={"val_rmse_before": 5.5, "val_rmse_after": 4.0, "innov_ms": 2.0}),
        _upd(delta=d2),
    ]
    for v in variants:
        assert v.signing_payload() != p
    assert base.with_signature(b"anything").signing_payload() == p


def test_bad_field_sizes_rejected() -> None:
    with pytest.raises(MessageError):
        GlobalBroadcast(round=0, nonce=b"short", params=np.zeros(3, np.float32))
    with pytest.raises(MessageError):
        _upd(model_hash=b"x")
    with pytest.raises(MessageError):
        _upd(delta=np.zeros((2, 2), np.float32))


def test_nonfinite_delta_survives_the_wire() -> None:
    """Messages carry NaN/inf faithfully; rejecting them is the server's job (and is logged)."""
    d = np.zeros(10, np.float32)
    d[2] = np.nan
    out = ClientUpdate.from_bytes(_upd(delta=d).to_bytes())
    assert np.isnan(out.delta[2])
