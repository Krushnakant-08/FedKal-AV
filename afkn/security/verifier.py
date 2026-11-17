"""Layer 1: is the sender an enrolled vehicle? Hard accept/reject, with a logged reason.

Checks, in this order (first failure wins)::

    unenrolled      id not in the registry (also: a valid signature from an unknown key)
    revoked         id is revoked
    bad_signature   signature invalid -> forged id, tampered payload or wrong key
    replay_round    update round != current round
    replay_nonce    update nonce != the nonce issued this round
    stale_model     model_hash != hash of the model the server broadcast this round
    duplicate       this vehicle already had an update accepted this round
    bad_shape       delta length != number of parameters
    non_finite      delta contains NaN or inf

An id is recorded as "seen" only when an update passes every check, so a forger cannot lock the real
vehicle out of its round. A hijacked vehicle that still holds a valid key passes Layer 1 by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from afkn.fed.messages import ClientUpdate, GlobalBroadcast
from afkn.security.registry import KeyRegistry
from afkn.security.signing import verify_signature

REASONS: tuple[str, ...] = (
    "unenrolled", "revoked", "bad_signature", "replay_round", "replay_nonce",
    "stale_model", "duplicate", "bad_shape", "non_finite",
)


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str | None  # None iff ok


class IdentityVerifier:
    """Stateful per round: call ``begin_round`` with each broadcast, then ``check`` every update."""

    def __init__(self, registry: KeyRegistry) -> None:
        self.registry = registry
        self._bc: GlobalBroadcast | None = None
        self._seen: set[int] = set()

    def begin_round(self, bc: GlobalBroadcast) -> None:
        self._bc = bc
        self._seen = set()

    def check(self, upd: ClientUpdate) -> CheckResult:
        bc = self._bc
        if bc is None:
            raise RuntimeError("begin_round must be called before check")
        vid = upd.client_id
        if not self.registry.is_enrolled(vid):
            return CheckResult(False, "unenrolled")
        if self.registry.is_revoked(vid):
            return CheckResult(False, "revoked")
        if not verify_signature(upd, self.registry.public_key(vid)):
            return CheckResult(False, "bad_signature")
        if upd.round != bc.round:
            return CheckResult(False, "replay_round")
        if upd.nonce != bc.nonce:
            return CheckResult(False, "replay_nonce")
        if upd.model_hash != bc.model_hash:
            return CheckResult(False, "stale_model")
        if vid in self._seen:
            return CheckResult(False, "duplicate")
        d = np.asarray(upd.delta)
        if d.shape != bc.params.shape:
            return CheckResult(False, "bad_shape")
        if not np.isfinite(d).all():
            return CheckResult(False, "non_finite")
        self._seen.add(vid)
        return CheckResult(True, None)
