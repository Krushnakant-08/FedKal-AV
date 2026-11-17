"""Differentiable PyTorch twin of ``afkn.sim.motion`` (same equations, same conventions).

Kept in lock-step with the NumPy version by ``tests/test_kalmannet.py`` (value parity in float64
and Jacobian parity through autograd), so the filter baselines and KalmanNet share one model.
"""

from __future__ import annotations

import math

import torch

MODELS: tuple[str, ...] = ("ctrv", "bicycle")


def f_torch(x: torch.Tensor, u: torch.Tensor, dt: float, model: str = "ctrv") -> torch.Tensor:
    """State transition for ``x (..., 4)`` and ``u (..., 2)``; broadcasts over leading dims."""
    if model not in MODELS:
        raise ValueError(f"unknown motion model '{model}', expected one of {MODELS}")
    px, py, theta, v = x.unbind(-1)
    w, a = u.unbind(-1)
    vm = v + 0.5 * a * dt  # mid-step speed
    y = 0.5 * w * dt
    thm = theta + y  # mid-step heading
    dist = vm * dt * torch.sinc(y / math.pi) if model == "ctrv" else vm * dt  # sinc(y/pi) = sin(y)/y
    return torch.stack([px + dist * torch.cos(thm), py + dist * torch.sin(thm), theta + w * dt, v + a * dt], dim=-1)


def h_torch(x: torch.Tensor) -> torch.Tensor:
    """GNSS measurement function: position only."""
    return x[..., :2]


def wrap_angle_torch(a: torch.Tensor) -> torch.Tensor:
    """Wrap to (-pi, pi]; differentiable almost everywhere."""
    return torch.atan2(torch.sin(a), torch.cos(a))
