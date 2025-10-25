"""Mathematical regression tests for optimisation and sampler components."""

from __future__ import annotations

from functools import partial
import numpy as np
import pytest

import chronopt as chron


def sphere(x: list[float]) -> np.ndarray:
    """Simple convex bowl with minimum at the origin."""
    arr = np.asarray(x, dtype=float)
    value = float(np.dot(arr, arr))
    return np.asarray([value], dtype=float)


def rosenbrock(x: list[float]) -> np.ndarray:
    """Classic Rosenbrock banana function with minimum at (1, 1)."""
    x1, x2 = x
    value = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    return np.asarray([value], dtype=float)


def booth(x: list[float]) -> np.ndarray:
    """Booth function with a known minimum at (1, 3)."""
    x1, x2 = x
    value = (x1 + 2 * x2 - 7) ** 2 + (2 * x1 + x2 - 5) ** 2
    return np.asarray([value], dtype=float)


def extended_rosenbrock(x: list[float]) -> np.ndarray:
    """N-dimensional Rosenbrock with narrow curved valley."""
    arr = np.asarray(x, dtype=float)
    value = np.sum(100 * (arr[1:] - arr[:-1] ** 2) ** 2 + (1 - arr[:-1]) ** 2)
    return np.asarray([value], dtype=float)


def matyas(x: list[float]) -> np.ndarray:
    """Matyas function - very flat near minimum at (0, 0)."""
    x1, x2 = x
    value = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    return np.asarray([value], dtype=float)


def rastrigin(x: list[float]) -> np.ndarray:
    """Rastrigin function - many local minima, global minimum at origin."""
    arr = np.asarray(x, dtype=float)
    n = len(arr)
    value = 10 * n + np.sum(arr**2 - 10 * np.cos(2 * np.pi * arr))
    return np.asarray([value], dtype=float)


def ridge(x: list[float], alpha: float = 1.0) -> np.ndarray:
    """Ridge function - nearly flat in perpendicular directions."""
    arr = np.asarray(x, dtype=float)
    value = arr[0] ** 2 + alpha * np.sum(arr[1:] ** 2)
    return np.asarray([value], dtype=float)


def make_nelder_mead() -> chron.NelderMead:
    return (
        chron.NelderMead()
        .with_max_iter(800)
        .with_threshold(1e-8)
        .with_position_tolerance(1e-6)
    )


def make_cmaes() -> chron.CMAES:
    return (
        chron.CMAES()
        .with_max_iter(1500)
        .with_threshold(1e-8)
        .with_sigma0(0.8)
        .with_patience(20.0)
        .with_seed(123)
    )


@pytest.mark.parametrize(
    "optimiser_factory",
    [
        pytest.param(make_nelder_mead, id="nelder-mead"),
        pytest.param(make_cmaes, id="cmaes"),
    ],
)
@pytest.mark.parametrize(
    "objective, dimension, initial, expected, position_tol, fun_tol",
    [
        pytest.param(
            sphere,
            3,
            np.array([3.5, -2.0, 0.5], dtype=float),
            np.zeros(3, dtype=float),
            1e-3,
            1e-8,
            id="sphere",
        ),
        pytest.param(
            rosenbrock,
            2,
            np.array([-1.5, 2.0], dtype=float),
            np.ones(2, dtype=float),
            5e-3,
            1e-6,
            id="rosenbrock",
        ),
        pytest.param(
            booth,
            2,
            np.array([0.0, 0.0], dtype=float),
            np.array([1.0, 3.0], dtype=float),
            1e-3,
            1e-8,
            id="booth",
        ),
        pytest.param(
            matyas, 2, np.array([10.0, 10.0]), np.zeros(2), 5e-2, 1e-6, id="matyas-flat"
        ),
        pytest.param(
            rastrigin,
            3,
            np.array([2.0, -2.0, 1.5]),
            np.zeros(3),
            0.5,
            5.0,
            id="rastrigin-multimodal",
            marks=pytest.mark.xfail,
        ),
        pytest.param(
            extended_rosenbrock,
            4,
            np.array([-1.0, 1.5, -0.5, 1.0]),
            np.ones(4),
            0.1,
            1e-4,
            id="rosenbrock-nd",
        ),
        pytest.param(
            partial(ridge, alpha=100.0),
            5,
            np.array([1.0, 2.0, -1.0, 0.5, -0.5]),
            np.zeros(5),
            1e-3,
            1e-5,
            id="ridge-identifiability",
        ),
    ],
)
def test_python_objectives_converge(
    optimiser_factory,
    objective,
    dimension,
    initial,
    expected,
    position_tol,
    fun_tol,
):
    """Ensure optimisation reaches known minima for several analytic functions."""

    builder = chron.PythonBuilder().add_callable(objective)
    for idx in range(dimension):
        builder = builder.add_parameter(f"x{idx}")

    problem = builder.build()

    optimiser = optimiser_factory()
    result = optimiser.run(problem, initial.tolist())

    assert result.success
    assert result.fun < fun_tol
    assert np.allclose(result.x, expected, atol=position_tol)


_LOGISTIC_DSL = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""


def _logistic_curve(t: np.ndarray, r: float, k: float, y0: float = 0.1) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return k / (1 + ((k / y0) - 1) * np.exp(-r * t))


def test_diffsol_logistic_convergence():
    """Validate Diffsol optimisation recovers parameters on logistic growth."""

    r_true, k_true = 1.0, 1.0
    time_points = np.arange(0, 10, dtype=float)
    data = _logistic_curve(time_points, r_true, k_true)

    builder = (
        chron.DiffsolBuilder()
        .add_diffsl(_LOGISTIC_DSL)
        .add_data(data)
        .with_rtol(1e-6)
        .add_params({"r": r_true, "k": k_true})
    )

    problem = builder.build()

    optimiser = (
        chron.NelderMead()
        .with_max_iter(400)
        .with_threshold(1e-7)
        .with_position_tolerance(1e-6)
    )
    result = optimiser.run(problem, [0.6, 1.4])

    assert result.success
    assert result.fun < 1e-3
    assert np.allclose(result.x, [r_true, k_true], atol=0.2)
