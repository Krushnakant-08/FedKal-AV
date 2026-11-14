"""Seeding: purity, independence, order-independence, cross-process stability."""

import hashlib

import numpy as np
import pytest

from afkn.seeding import SeedTree


def test_same_path_same_stream():
    a = SeedTree(7).child("client", 3, "train").rng().standard_normal(50)
    b = SeedTree(7).child("client", 3, "train").rng().standard_normal(50)
    assert np.array_equal(a, b)


def test_different_seed_different_stream():
    a = SeedTree(7).child("x").rng().standard_normal(50)
    b = SeedTree(8).child("x").rng().standard_normal(50)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize(
    "p, q",
    [
        (("a", 1), ("a", 2)),
        (("a",), ("b",)),
        ((1,), ("1",)),  # int and str keys never collide
        (("a", "b"), ("a",)),
        ((), ("a",)),
    ],
)
def test_distinct_paths_distinct_streams(p, q):
    a = SeedTree(0).child(*p).rng().standard_normal(20)
    b = SeedTree(0).child(*q).rng().standard_normal(20)
    assert not np.array_equal(a, b)


def test_creation_order_does_not_matter():
    root = SeedTree(5)
    first = root.child("c", 1).rng().standard_normal(10)
    _ = root.child("c", 2).rng().standard_normal(1000)  # consume a sibling
    again = root.child("c", 1).rng().standard_normal(10)
    assert np.array_equal(first, again)


def test_child_is_composable():
    a = SeedTree(1).child("a").child("b", 2)
    b = SeedTree(1).child("a", "b", 2)
    assert a == b


def test_spawn_key_is_process_stable():
    # String keys are hashed with SHA-256 (not Python's salted hash()).
    h = hashlib.sha256(b"client").digest()
    expected = (1, int.from_bytes(h[:4], "big"), int.from_bytes(h[4:8], "big"), 0, 3)
    assert SeedTree(0).child("client", 3).spawn_key() == expected


def test_int_seed_deterministic_and_bounded():
    s1 = SeedTree(3).child("torch").int_seed()
    s2 = SeedTree(3).child("torch").int_seed()
    assert s1 == s2 and 0 <= s1 < 2**63
    assert SeedTree(3).child("torch").int_seed() != SeedTree(3).child("python").int_seed()
    assert 0 <= SeedTree(3).int_seed(bits=31) < 2**31


@pytest.mark.parametrize("bad", [True, 1.5, -1, 2**32, None])
def test_invalid_keys_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        SeedTree(0).child(bad)


def test_invalid_entropy_rejected():
    with pytest.raises((TypeError, ValueError)):
        SeedTree(-1)
    with pytest.raises(TypeError):
        SeedTree(1.5)  # type: ignore[arg-type]
