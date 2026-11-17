"""Byte-level wire messages between server and clients (torch-free, deterministic).

Layouts (big-endian header fields, little-endian float32 payloads)::

    GlobalBroadcast  b"AFKB" | round u32 | nonce 16s | model_hash 32s | n_params u32 | params f32[n]
    ClientUpdate     b"AFKU" | client_id u32 | round u32 | nonce 16s | model_hash 32s | n_samples u32
                     | n_signals u16 | (name_len u8, name, value f64)* | sig_len u16 | signature
                     | n_params u32 | delta f32[n]

``model_hash`` = SHA-256 of the broadcast parameters as little-endian float32 bytes.
``signing_payload()`` is what Step 5 signs: every field except the signature, with the delta
replaced by its SHA-256 (``delta_hash``), so tampering with any field or the delta breaks it.

Communication accounting follows the project rule: parameters x 4 bytes per direction per
participating client (``payload_bytes``); ``wire_bytes`` adds the (small) header overhead and is
logged separately. Parsing is strict: bad magic, truncation and trailing bytes raise ``MessageError``.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

import numpy as np

NONCE_LEN = 16
HASH_LEN = 32
_MAGIC_B = b"AFKB"
_MAGIC_U = b"AFKU"


class MessageError(ValueError):
    """Malformed wire message."""


def params_bytes(params: np.ndarray) -> bytes:
    """Canonical little-endian float32 bytes of a 1-D parameter/delta vector."""
    return np.ascontiguousarray(np.asarray(params), dtype="<f4").tobytes()


def hash_params(params: np.ndarray) -> bytes:
    """Raw 32-byte SHA-256 of the canonical parameter bytes."""
    return hashlib.sha256(params_bytes(params)).digest()


def _check_fixed(name: str, value: bytes, n: int) -> None:
    if not isinstance(value, (bytes, bytearray)) or len(value) != n:
        raise MessageError(f"{name} must be {n} bytes")


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf, self.pos = bytes(buf), 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise MessageError("truncated message")
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str) -> tuple:
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))

    def done(self) -> None:
        if self.pos != len(self.buf):
            raise MessageError("trailing bytes")


@dataclass(frozen=True, eq=False)
class GlobalBroadcast:
    """Server -> client: current global model for ``round`` with its nonce and hash."""

    round: int
    nonce: bytes
    params: np.ndarray  # (P,) float32

    def __post_init__(self) -> None:
        _check_fixed("nonce", self.nonce, NONCE_LEN)
        if np.asarray(self.params).ndim != 1:
            raise MessageError("params must be 1-D")

    @property
    def model_hash(self) -> bytes:
        return hash_params(self.params)

    @property
    def payload_bytes(self) -> int:
        return int(np.asarray(self.params).size) * 4

    def to_bytes(self) -> bytes:
        p = np.asarray(self.params)
        return (
            _MAGIC_B
            + struct.pack(">I", self.round)
            + self.nonce
            + self.model_hash
            + struct.pack(">I", p.size)
            + params_bytes(p)
        )

    @property
    def wire_bytes(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> GlobalBroadcast:
        r = _Reader(buf)
        if r.take(4) != _MAGIC_B:
            raise MessageError("bad magic")
        (rnd,) = r.unpack(">I")
        nonce, mh = r.take(NONCE_LEN), r.take(HASH_LEN)
        (n,) = r.unpack(">I")
        params = np.frombuffer(r.take(4 * n), dtype="<f4").astype(np.float32)
        r.done()
        msg = cls(rnd, nonce, params)
        if msg.model_hash != mh:
            raise MessageError("model_hash does not match params")
        return msg


@dataclass(frozen=True, eq=False)
class ClientUpdate:
    """Client -> server: parameter delta plus the client's self-reported signals."""

    client_id: int
    round: int
    nonce: bytes
    model_hash: bytes  # hash of the broadcast this delta was trained from
    delta: np.ndarray  # (P,) float32
    n_samples: int
    signals: dict[str, float] = field(default_factory=dict)
    signature: bytes = b""

    def __post_init__(self) -> None:
        _check_fixed("nonce", self.nonce, NONCE_LEN)
        _check_fixed("model_hash", self.model_hash, HASH_LEN)
        if np.asarray(self.delta).ndim != 1:
            raise MessageError("delta must be 1-D")
        if len(self.signature) > 0xFFFF:
            raise MessageError("signature too long")

    @property
    def delta_hash(self) -> bytes:
        return hash_params(self.delta)

    @property
    def payload_bytes(self) -> int:
        return int(np.asarray(self.delta).size) * 4

    def _head(self) -> bytes:
        out = [
            struct.pack(">II", self.client_id, self.round),
            self.nonce,
            self.model_hash,
            struct.pack(">IH", self.n_samples, len(self.signals)),
        ]
        for name in sorted(self.signals):
            raw = name.encode("utf-8")
            if not 0 < len(raw) < 256:
                raise MessageError("signal name must be 1..255 bytes")
            out += [struct.pack(">B", len(raw)), raw, struct.pack(">d", float(self.signals[name]))]
        return b"".join(out)

    def signing_payload(self) -> bytes:
        """Bytes covered by the signature (everything but the signature; delta via its hash)."""
        return b"AFKU-SIGN" + self._head() + self.delta_hash

    def with_signature(self, signature: bytes) -> ClientUpdate:
        return ClientUpdate(self.client_id, self.round, self.nonce, self.model_hash, self.delta,
                            self.n_samples, dict(self.signals), bytes(signature))

    def to_bytes(self) -> bytes:
        d = np.asarray(self.delta)
        return (
            _MAGIC_U
            + self._head()
            + struct.pack(">H", len(self.signature))
            + self.signature
            + struct.pack(">I", d.size)
            + params_bytes(d)
        )

    @property
    def wire_bytes(self) -> int:
        return len(self.to_bytes())

    @classmethod
    def from_bytes(cls, buf: bytes) -> ClientUpdate:
        r = _Reader(buf)
        if r.take(4) != _MAGIC_U:
            raise MessageError("bad magic")
        cid, rnd = r.unpack(">II")
        nonce, mh = r.take(NONCE_LEN), r.take(HASH_LEN)
        n_samples, n_sig = r.unpack(">IH")
        signals: dict[str, float] = {}
        for _ in range(n_sig):
            (ln,) = r.unpack(">B")
            name = r.take(ln).decode("utf-8")
            (val,) = r.unpack(">d")
            if name in signals:
                raise MessageError("duplicate signal name")
            signals[name] = val
        (sl,) = r.unpack(">H")
        sig = r.take(sl)
        (n,) = r.unpack(">I")
        delta = np.frombuffer(r.take(4 * n), dtype="<f4").astype(np.float32)
        r.done()
        return cls(cid, rnd, nonce, mh, delta, n_samples, signals, sig)
