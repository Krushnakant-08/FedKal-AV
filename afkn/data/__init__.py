"""Federated dataset construction (per-client splits, global test, server-trusted set)."""

from afkn.data.partition import (
    ClientData,
    FederatedDataset,
    build_client_data,
    build_federated_dataset,
    build_mixed_set,
)

__all__ = [
    "ClientData",
    "FederatedDataset",
    "build_client_data",
    "build_federated_dataset",
    "build_mixed_set",
]
