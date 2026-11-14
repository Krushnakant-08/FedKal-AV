"""Motion models: Jacobians vs finite differences, exactness, limits, shapes."""

from __future__ import annotations

import numpy as np
import pytest

from afkn.sim.motion import H, f, h, jac_u, jac_x, steering_angle, wrap_angle

DT = 0.1
MODELS = ("ctrv", "bicycle")
YAW_RATES = (0.0, 1e-12, -1e-9, 1e-6, -3e-3, 0.05, 0.4, -1.5)  # spans the singular limit


def _fd_jacobians(x, u, dt, model, eps=1e-6):
    jx = np.zeros((4, 4))
    ju = np.zeros((4, 2))
    for j in range(4):
        d = np.zeros(4)
        d[j] = eps
        jx[:, j] = (f(x + d, u, dt, model) - f(x - d, u, dt, model)) / (2 * eps)
    for j in range(2):
        d = np.zeros(2)
        d[j] = eps
        ju[:, j] = (f(x, u + d, dt, model) - f(x, u - d, dt, model)) / (2 * eps)
    return jx, ju


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("w", YAW_RATES)
def test_analytic_jacobians_match_finite_differences(model, w):
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = np.array([rng.normal(0, 30), rng.normal(0, 30), rng.uniform(-3, 3), rng.uniform(2, 30)])
        u = np.array([w, rng.uniform(-2, 2)])
        jx_fd, ju_fd = _fd_jacobians(x, u, DT, model)
        assert np.allclose(jac_x(x, u, DT, model), jx_fd, atol=1e-6, rtol=1e-6)
        assert np.allclose(jac_u(x, u, DT, model), ju_fd, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("w", [0.0, 1e-12, 1e-4, 0.3])
def test_ctrv_is_exact_arc_composition(w):
    """Exact CTRV: two half-steps equal one full step (a = 0)."""
    x = np.array([3.0, -2.0, 0.7, 12.0])
    u = np.array([w, 0.0])
    full = f(x, u, 1.0, "ctrv")
    half = f(f(x, u, 0.5, "ctrv"), u, 0.5, "ctrv")
    assert np.allclose(full, half, atol=1e-12)


def test_ctrv_matches_closed_form_arc():
    x0 = np.array([0.0, 0.0, 0.0, 10.0])
    w = 0.5
    out = f(x0, np.array([w, 0.0]), 2.0, "ctrv")
    v = 10.0
    assert np.allclose(out[:2], [v / w * np.sin(w * 2.0), v / w * (1 - np.cos(w * 2.0))], atol=1e-12)
    assert np.isclose(out[2], w * 2.0) and np.isclose(out[3], v)


def test_bicycle_is_first_order_and_converges_to_ctrv():
    """Bicycle midpoint-Euler differs from the exact arc by O(dt^3) per step."""
    x = np.array([0.0, 0.0, 0.3, 15.0])
    u = np.array([0.8, 0.0])

    def err(dt):
        return np.linalg.norm(f(x, u, dt, "bicycle") - f(x, u, dt, "ctrv"))

    assert err(0.5) > 1e-5  # models are genuinely different
    ratio = err(0.4) / err(0.2)
    assert 6.5 < ratio < 9.5  # ~ 2^3


@pytest.mark.parametrize("model", MODELS)
def test_straight_line_and_constant_acceleration(model):
    x = np.array([1.0, 2.0, 0.0, 10.0])
    out = f(x, np.array([0.0, 2.0]), DT, model)
    assert np.allclose(out, [1.0 + 10.0 * DT + 0.5 * 2.0 * DT**2, 2.0, 0.0, 10.0 + 2.0 * DT], atol=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_continuous_through_zero_yaw_rate(model):
    x = np.array([5.0, 5.0, 1.0, 20.0])
    ref = f(x, np.array([0.0, 0.5]), DT, model)
    for w in (1e-12, 1e-8, -1e-8):
        assert np.allclose(f(x, np.array([w, 0.5]), DT, model), ref, atol=1e-6)


def test_models_agree_at_zero_yaw_rate():
    x = np.array([1.0, 1.0, 0.4, 8.0])
    u = np.array([0.0, 1.0])
    assert np.allclose(f(x, u, DT, "ctrv"), f(x, u, DT, "bicycle"), atol=1e-12)


@pytest.mark.parametrize("model", MODELS)
def test_heading_is_not_wrapped(model):
    x = np.array([0.0, 0.0, 3.1, 5.0])
    out = f(x, np.array([1.0, 0.0]), DT, model)
    assert out[2] > np.pi  # the stored heading continues past pi


@pytest.mark.parametrize("model", MODELS)
def test_broadcasting_shapes(model):
    x = np.random.default_rng(1).normal(size=(7, 5, 4))
    u = np.random.default_rng(2).normal(size=(7, 5, 2))
    assert f(x, u, DT, model).shape == (7, 5, 4)
    assert jac_x(x, u, DT, model).shape == (7, 5, 4, 4)
    assert jac_u(x, u, DT, model).shape == (7, 5, 4, 2)
    # batched result equals the per-sample result
    assert np.allclose(f(x, u, DT, model)[3, 2], f(x[3, 2], u[3, 2], DT, model))
    # (N,4) states with a shared (2,) input broadcast
    assert f(x[:, 0], u[0, 0], DT, model).shape == (7, 4)


def test_measurement_model():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    assert np.array_equal(h(x), [[1.0, 2.0], [5.0, 6.0]])
    assert np.array_equal(H @ x[0], h(x[0]))


def test_wrap_angle_and_steering():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-np.pi), np.pi)
    assert np.allclose(wrap_angle(np.array([0.0, 2 * np.pi])), 0.0, atol=1e-12)
    assert np.isclose(steering_angle(0.2, 10.0, 2.7), np.arctan(0.054))


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        f(np.zeros(4), np.zeros(2), DT, "segway")
    with pytest.raises(ValueError):
        f(np.zeros(3), np.zeros(2), DT, "ctrv")
