"""Hierarchical, order-independent seeding built on ``numpy.random.SeedSequence``.

Every random source in the project is a pure function of ``(root seed, path)``.
A :class:`SeedTree` is just that pair; it holds no mutable state, so creating or
using one stream can never perturb another (no global RNG, no consumption-order
coupling). The same seed therefore yields bit-identical data.

Example::

    root = SeedTree(42)
    rng = root.child("client", 3, "train", "imu").rng()

Calling ``.rng()`` twice on the same node returns two *independent objects that
produce the same stream*. That is intentional (purity); give every consumer its
own named child.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

Key = int | str

_U32 = 2**32


def _encode_key(key: Key) -> tuple[int, ...]:
    """Encode one path element as 32-bit words, prefix-free and type-tagged.

    ``int k`` -> ``(0, k)``; ``str s`` -> ``(1, hi32, lo32)`` of SHA-256(s).
    The tag makes ``1`` and ``"1"`` distinct, and the encoding is stable across
    processes (unlike Python's salted ``hash``).
    """
    if isinstance(key, bool):
        raise TypeError("bool is not a valid seed key")
    if isinstance(key, (int, np.integer)):
        k = int(key)
        if not 0 <= k < _U32:
            raise ValueError(f"integer key must be in [0, 2**32), got {k}")
        return (0, k)
    if isinstance(key, str):
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return (1, int.from_bytes(digest[:4], "big"), int.from_bytes(digest[4:8], "big"))
    raise TypeError(f"seed key must be int or str, got {type(key).__name__}")


@dataclass(frozen=True)
class SeedTree:
    """Immutable node in a seed hierarchy: ``(entropy, path)``."""

    entropy: int
    path: tuple[Key, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.entropy, bool) or not isinstance(self.entropy, (int, np.integer)):
            raise TypeError("entropy must be an int")
        if int(self.entropy) < 0:
            raise ValueError("entropy must be non-negative")

    def child(self, *keys: Key) -> "SeedTree":
        """Return the descendant node at ``path + keys``."""
        for k in keys:
            _encode_key(k)  # validate eagerly
        return SeedTree(int(self.entropy), self.path + tuple(keys))

    def spawn_key(self) -> tuple[int, ...]:
        """The flat tuple of 32-bit words passed to ``SeedSequence``."""
        out: list[int] = []
        for k in self.path:
            out.extend(_encode_key(k))
        return tuple(out)

    def sequence(self) -> np.random.SeedSequence:
        return np.random.SeedSequence(entropy=int(self.entropy), spawn_key=self.spawn_key())

    def rng(self) -> np.random.Generator:
        """A fresh PCG64 generator for this node (same stream on every call)."""
        return np.random.Generator(np.random.PCG64(self.sequence()))

    def int_seed(self, bits: int = 63) -> int:
        """A deterministic integer seed (e.g. for ``torch.manual_seed``/``random.seed``)."""
        if not 1 <= bits <= 64:
            raise ValueError("bits must be in [1, 64]")
        lo, hi = (int(v) for v in self.sequence().generate_state(2, dtype=np.uint32))
        return ((hi << 32) | lo) & ((1 << bits) - 1)
