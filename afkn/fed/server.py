"""Federated server: broadcasts, structural checks, weighted aggregation. Step 5 adds Ed25519 checks."""

from __future__ import annotations

import numpy as np

from afkn.fed.aggregate import apply_deltas, normalize_weights
from afkn.fed.config import FedSpec
from afkn.fed.messages import NONCE_LEN, ClientUpdate, GlobalBroadcast
from afkn.seeding import SeedTree

# afkn.security imports afkn.fed.messages only, so there is no import cycle.
from afkn.security.verifier import IdentityVerifier


class Server:
    """Holds the global parameters. The nonce is derived from the seed (simulation determinism);
    a deployment would draw it from a CSPRNG."""

    def __init__(self, init_params: np.ndarray, fed: FedSpec, tree: SeedTree,
                 verifier: IdentityVerifier | None = None) -> None:
        self.params = np.asarray(init_params, dtype=np.float32).copy()
        self.fed = fed
        self.tree = tree
        self.verifier = verifier  # Layer 1 (Ed25519 identity checks); None = structural checks only

    def broadcast(self, round_idx: int) -> GlobalBroadcast:
        nonce = self.tree.child("nonce", round_idx).rng().bytes(NONCE_LEN)
        bc = GlobalBroadcast(round=round_idx, nonce=nonce, params=self.params.copy())
        if self.verifier is not None:
            self.verifier.begin_round(bc)
        return bc

    def check_update(self, upd: ClientUpdate, bc: GlobalBroadcast) -> str | None:
        """Reject reason or None. Uses the Layer 1 verifier when configured, else ``basic_checks``."""
        if self.verifier is None:
            return self.basic_checks(upd, bc)
        return self.verifier.check(upd).reason

    def basic_checks(self, upd: ClientUpdate, bc: GlobalBroadcast) -> str | None:
        """Structural sanity only (reason string, or None if fine). The cryptographic identity checks
        (signature, revocation, duplicate sender) are Step 5; rejections are always reported, never silent."""
        if upd.round != bc.round:
            return "round_mismatch"
        if upd.nonce != bc.nonce:
            return "nonce_mismatch"
        if upd.model_hash != bc.model_hash:
            return "model_hash_mismatch"
        d = np.asarray(upd.delta)
        if d.shape != self.params.shape:
            return "bad_shape"
        if not np.isfinite(d).all():
            return "non_finite"
        return None

    def final_weights(self, raw: np.ndarray) -> np.ndarray:
        """Raw scores -> weights summing to 1 with the configured floor enforced."""
        return normalize_weights(raw, self.fed.weight_floor)

    def aggregate(self, updates: list[ClientUpdate], weights: np.ndarray) -> None:
        """Apply the weighted average of the (already accepted) deltas to the global model."""
        if not updates:
            return  # nothing accepted this round: model unchanged (the loop logs this)
        deltas = np.stack([np.asarray(u.delta) for u in updates])
        self.params = apply_deltas(self.params, deltas, weights, self.fed.server_lr)
