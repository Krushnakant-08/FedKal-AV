"""Ed25519 vehicle identities (via ``cryptography``).

Keys are generated at runtime and never written to disk. For reproducible simulations a key can be
derived from a SeedTree stream: that is a SIMULATION convenience only (a real vehicle generates its key
in a secure element; hardware attestation is out of scope and cannot be simulated here).
No formal security guarantee is claimed: these are unit-tested identity checks in a simulation.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from afkn.fed.messages import ClientUpdate
from afkn.seeding import SeedTree

SIGNATURE_LEN = 64
PUBLIC_KEY_LEN = 32


def generate_private_key(tree: SeedTree | None = None) -> Ed25519PrivateKey:
    """Fresh random key, or (simulation only) a key derived deterministically from ``tree``."""
    if tree is None:
        return Ed25519PrivateKey.generate()
    return Ed25519PrivateKey.from_private_bytes(tree.rng().bytes(32))


def public_bytes(key: Ed25519PrivateKey | Ed25519PublicKey) -> bytes:
    """Raw 32-byte public key."""
    pub = key.public_key() if isinstance(key, Ed25519PrivateKey) else key
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def sign_update(update: ClientUpdate, key: Ed25519PrivateKey) -> ClientUpdate:
    """Return ``update`` carrying an Ed25519 signature over ``update.signing_payload()``."""
    return update.with_signature(key.sign(update.signing_payload()))


def verify_signature(update: ClientUpdate, public_key: bytes) -> bool:
    """True iff the signature is valid for this exact update under ``public_key``."""
    if len(update.signature) != SIGNATURE_LEN or len(public_key) != PUBLIC_KEY_LEN:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(update.signature, update.signing_payload())
    except (InvalidSignature, ValueError):
        return False
    return True
