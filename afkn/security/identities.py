"""Fleet identities for simulations: one runtime-generated key per client, plus the registry."""

from __future__ import annotations

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from afkn.fed.messages import ClientUpdate
from afkn.security.registry import KeyRegistry
from afkn.security.signing import generate_private_key, public_bytes, sign_update
from afkn.seeding import SeedTree


class Identities:
    """Holds every enrolled client's private key (simulation only; keys exist only in memory)."""

    def __init__(self, n_clients: int, tree: SeedTree) -> None:
        self.keys: dict[int, Ed25519PrivateKey] = {i: generate_private_key(tree.child("key", i)) for i in range(n_clients)}
        self.registry = KeyRegistry()
        for i, k in self.keys.items():
            self.registry.enroll(i, public_bytes(k))

    def sign(self, upd: ClientUpdate) -> ClientUpdate:
        return sign_update(upd, self.keys[upd.client_id])
