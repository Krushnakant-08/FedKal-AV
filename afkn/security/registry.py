"""Enrollment registry: ``vehicle_id -> public key`` with a revoked flag."""

from __future__ import annotations

from afkn.security.signing import PUBLIC_KEY_LEN


class RegistryError(ValueError):
    """Invalid enrollment or revocation request."""


class KeyRegistry:
    """Server-side registry. Enrolling the same id twice is an error (no silent key replacement)."""

    def __init__(self) -> None:
        self._keys: dict[int, bytes] = {}
        self._revoked: set[int] = set()

    def enroll(self, vehicle_id: int, public_key: bytes) -> None:
        if len(public_key) != PUBLIC_KEY_LEN:
            raise RegistryError(f"public key must be {PUBLIC_KEY_LEN} bytes")
        if vehicle_id in self._keys:
            raise RegistryError(f"vehicle {vehicle_id} is already enrolled")
        self._keys[vehicle_id] = bytes(public_key)

    def revoke(self, vehicle_id: int) -> None:
        if vehicle_id not in self._keys:
            raise RegistryError(f"vehicle {vehicle_id} is not enrolled")
        self._revoked.add(vehicle_id)

    def is_enrolled(self, vehicle_id: int) -> bool:
        return vehicle_id in self._keys

    def is_revoked(self, vehicle_id: int) -> bool:
        return vehicle_id in self._revoked

    def public_key(self, vehicle_id: int) -> bytes:
        return self._keys[vehicle_id]

    def __len__(self) -> int:
        return len(self._keys)
