"""Motion models shared by the simulator, the filters (Step 2) and KalmanNet (Step 3).

State  x = [px, py, heading, speed]
Input  u = [yaw_rate, longitudinal_acceleration]
Meas.  z = h(x) = [px, py]

All functions broadcast over leading dimensions: ``x`` has shape ``(..., 4)`` and
``u`` has shape ``(..., 2)``.

Models (``model=``)
-------------------
``"ctrv"``     Constant turn rate and velocity: the *exact* circular-arc
               integration for constant yaw rate. Acceleration is handled with the
               mid-step speed ``v + a*dt/2`` (exact for a == 0, and exact for the
               travelled distance along a straight line for any a).
``"bicycle"``  Kinematic bicycle (rear-axle reference) driven by a yaw-rate input,
               discretised with midpoint-heading Euler. With yaw rate as the input
               the wheelbase drops out of the equations (steering angle is implied,
               ``delta = atan(L*omega/v)``); the wheelbase is used only to bound
               the planned yaw rate (see ``generator.plan_controls``). It differs
               from CTRV by an O(omega^2 dt^3) per-step arc-vs-chord term.

Both are written as ``p' = p + (v+a dt/2) dt S(omega dt/2) [cos, sin](theta + omega dt/2)``
with ``S(y) = sin(y)/y`` for CTRV and ``S = 1`` for the bicycle, so there is no
branch (and no singularity) at omega -> 0. Heading is never wrapped here; use
``wrap_angle`` on *differences* of headings.
"""

from __future__ import annotations

import numpy as np

MODELS: tuple[str, ...] = ("ctrv", "bicycle")
STATE_DIM = 4
MEAS_DIM = 2
CONTROL_DIM = 2

#: Constant measurement Jacobian dh/dx for h(x) = [px, py].
H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])


def _sinc(y: np.ndarray) -> np.ndarray:
    """sin(y)/y, smooth through y = 0."""
    return np.sinc(np.asarray(y, dtype=float) / np.pi)


def _dsinc(y: np.ndarray) -> np.ndarray:
    """d/dy [sin(y)/y] = (y cos y - sin y) / y^2, with a series near 0 to avoid cancellation."""
    y = np.asarray(y, dtype=float)
    small = np.abs(y) < 1e-2
    ys = np.where(small, 1.0, y)
    exact = (ys * np.cos(ys) - np.sin(ys)) / ys**2
    series = -y / 3.0 + y**3 / 30.0 - y**5 / 840.0
    return np.where(small, series, exact)


def _terms(x: np.ndarray, u: np.ndarray, dt: float, model: str):
    if model not in MODELS:
        raise ValueError(f"unknown motion model '{model}', expected one of {MODELS}")
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    if x.shape[-1] != STATE_DIM or u.shape[-1] != CONTROL_DIM:
        raise ValueError(f"expected x[..., {STATE_DIM}] and u[..., {CONTROL_DIM}], got {x.shape}, {u.shape}")
    theta, v = x[..., 2], x[..., 3]
    w, a = u[..., 0], u[..., 1]
    vm = v + 0.5 * a * dt  # mid-step speed
    y = 0.5 * w * dt
    thm = theta + y  # mid-step heading
    if model == "ctrv":
        s, ds_dw = _sinc(y), _dsinc(y) * 0.5 * dt
    else:
        s, ds_dw = np.ones_like(y), np.zeros_like(y)
    return x, u, vm, thm, s, ds_dw


def f(x: np.ndarray, u: np.ndarray, dt: float, model: str = "ctrv") -> np.ndarray:
    """Noise-free state transition ``x_{t+1} = f(x_t, u_t)``."""
    x, u, vm, thm, s, _ = _terms(x, u, dt, model)
    d = vm * dt * s
    comps = (
        x[..., 0] + d * np.cos(thm),
        x[..., 1] + d * np.sin(thm),
        x[..., 2] + u[..., 0] * dt,
        x[..., 3] + u[..., 1] * dt,
    )
    return np.stack(np.broadcast_arrays(*comps), axis=-1)


def jac_x(x: np.ndarray, u: np.ndarray, dt: float, model: str = "ctrv") -> np.ndarray:
    """Analytic state Jacobian F = df/dx, shape ``(..., 4, 4)``."""
    x, u, vm, thm, s, _ = _terms(x, u, dt, model)
    shape = np.broadcast_shapes(x.shape[:-1], u.shape[:-1])
    jac = np.zeros(shape + (4, 4))
    for i in range(4):
        jac[..., i, i] = 1.0
    jac[..., 0, 2] = -vm * dt * s * np.sin(thm)
    jac[..., 0, 3] = dt * s * np.cos(thm)
    jac[..., 1, 2] = vm * dt * s * np.cos(thm)
    jac[..., 1, 3] = dt * s * np.sin(thm)
    return jac


def jac_u(x: np.ndarray, u: np.ndarray, dt: float, model: str = "ctrv") -> np.ndarray:
    """Analytic input Jacobian G = df/du, shape ``(..., 4, 2)``."""
    x, u, vm, thm, s, ds_dw = _terms(x, u, dt, model)
    shape = np.broadcast_shapes(x.shape[:-1], u.shape[:-1])
    jac = np.zeros(shape + (4, 2))
    cos, sin = np.cos(thm), np.sin(thm)
    jac[..., 0, 0] = vm * dt * (-sin * 0.5 * dt * s + cos * ds_dw)
    jac[..., 0, 1] = 0.5 * dt * dt * s * cos
    jac[..., 1, 0] = vm * dt * (cos * 0.5 * dt * s + sin * ds_dw)
    jac[..., 1, 1] = 0.5 * dt * dt * s * sin
    jac[..., 2, 0] = dt
    jac[..., 3, 1] = dt
    return jac


def h(x: np.ndarray) -> np.ndarray:
    """GNSS measurement function: position only."""
    return np.asarray(x, dtype=float)[..., :2]


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]. Use on heading *differences*, never on stored headings."""
    return np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2.0 * np.pi)


def steering_angle(yaw_rate: np.ndarray, speed: np.ndarray, wheelbase: float) -> np.ndarray:
    """Front-wheel angle implied by a yaw rate at a given speed: atan(L * omega / v)."""
    return np.arctan(wheelbase * np.asarray(yaw_rate, dtype=float) / np.maximum(np.asarray(speed, dtype=float), 1e-6))
