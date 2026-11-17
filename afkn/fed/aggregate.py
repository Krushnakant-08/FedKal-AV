"""Aggregation primitives (torch-free): weights, weight floor, participant sampling, global update."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def fedavg_weights(n_samples: np.ndarray | list[int]) -> np.ndarray:
    """Classic FedAvg weights ``n_i / sum n``."""
    n = np.asarray(n_samples, dtype=np.float64)
    if n.ndim != 1 or n.size == 0 or (n < 0).any() or n.sum() <= 0:
        raise ValueError("n_samples must be a non-empty 1-D array of non-negative values with positive sum")
    return n / n.sum()


def normalize_weights(raw: np.ndarray | list[float], floor: float = 0.0) -> np.ndarray:
    """Turn non-negative raw scores into weights that sum to 1 with every weight >= ``floor``.

    ``w = floor + (1 - k*floor) * raw / sum(raw)``: the floor holds exactly after normalisation
    (no second renormalisation can push a weight below it). All-zero raw scores give uniform weights.
    """
    r = np.asarray(raw, dtype=np.float64)
    k = r.size
    if r.ndim != 1 or k == 0 or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("raw weights must be a non-empty 1-D array of finite non-negative values")
    if floor < 0.0 or floor * k > 1.0:
        raise ValueError(f"floor {floor} infeasible for {k} clients (floor * k must be <= 1)")
    s = r.sum()
    base = np.full(k, 1.0 / k) if s <= 0 else r / s
    return floor + (1.0 - k * floor) * base


def sample_participants(n_clients: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Sorted client ids for one round: ``max(1, round(fraction * n))`` sampled without replacement."""
    if n_clients < 1 or not 0.0 < fraction <= 1.0:
        raise ValueError("need n_clients >= 1 and 0 < fraction <= 1")
    k = max(1, round(fraction * n_clients))
    return np.sort(rng.choice(n_clients, size=k, replace=False))


def apply_deltas(global_params: np.ndarray, deltas: np.ndarray, weights: np.ndarray, server_lr: float = 1.0) -> np.ndarray:
    """``global + server_lr * sum_i w_i * delta_i`` in float64, returned as float32.

    Raises on shape mismatch, non-finite deltas or weights that are negative / do not sum to 1.
    """
    g = np.asarray(global_params)
    d = np.asarray(deltas, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != g.size or w.shape != (d.shape[0],):
        raise ValueError("shape mismatch between global params, deltas and weights")
    if not np.isfinite(d).all():
        raise ValueError("deltas contain NaN or inf")
    if (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-9):
        raise ValueError("weights must be non-negative and sum to 1")
    return (g.astype(np.float64) + server_lr * (w[:, None] * d).sum(axis=0)).astype(np.float32)


@dataclass(frozen=True)
class RoundContext:
    """What a weighting policy may know about the current round (server side only)."""

    round: int
    global_params: np.ndarray  # (P,) float32, the model broadcast this round


@dataclass(frozen=True)
class Weighting:
    """Result of a weighting policy for the accepted updates of one round.

    ``raw``: non-negative scores (normalised by the loop with ``floor``); ``info``: per-client log rows
    keyed by client id; ``deltas``: optional replacement deltas (e.g. norm-clipped), shape (k, P);
    ``floor``: overrides ``FedSpec.weight_floor`` when not None.
    """

    raw: np.ndarray
    info: dict[int, dict] = field(default_factory=dict)
    deltas: np.ndarray | None = None
    floor: float | None = None
