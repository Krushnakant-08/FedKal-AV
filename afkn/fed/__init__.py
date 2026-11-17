"""Federated learning: byte-level messages, aggregation, client, server and the round loop.

Torch-free parts are re-exported here. Import ``afkn.fed.client``, ``afkn.fed.server`` and
``afkn.fed.loop`` explicitly (they need PyTorch).
"""

from afkn.fed.aggregate import (
    apply_deltas,
    fedavg_weights,
    normalize_weights,
    sample_participants,
)
from afkn.fed.config import FedSpec, load_fed_config
from afkn.fed.messages import ClientUpdate, GlobalBroadcast, MessageError, hash_params

__all__ = [
    "ClientUpdate",
    "FedSpec",
    "GlobalBroadcast",
    "MessageError",
    "apply_deltas",
    "fedavg_weights",
    "hash_params",
    "load_fed_config",
    "normalize_weights",
    "sample_participants",
]
