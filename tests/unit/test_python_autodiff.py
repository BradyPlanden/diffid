import diffid
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy


def _quadratic_core(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sum(x**2)


def _quadratic_objective(x):
    return float(_quadratic_core(x))


_raw_grad = jax.grad(_quadratic_core)


def _quadratic_gradient(x):
    x = jnp.asarray(x, dtype=jnp.float32)
    return np.asarray(_raw_grad(x), dtype=np.float32)


def quadratic_problem():
    """Creates a 3D quadratic optimisation problem using ScalarBuilder"""
    return (
        diffid.ScalarBuilder()
        .with_objective(_quadratic_objective)
        .with_gradient(_quadratic_gradient)
        .with_parameter("x1", 1.0)
        .with_parameter("x2", 2.0)
        .with_parameter("x3", 3.0)
        .build()
    )


def test_python_builder_gradient_with_jax():
    """Test gradient computation with JAX"""
    x0 = np.array([1.5, -2.0, 0.25], dtype=np.float32)

    value = quadratic_problem().evaluate(x0.astype(float).tolist())
    assert np.isclose(value, float(np.dot(x0, x0)))

    gradient = quadratic_problem().evaluate_gradient(x0.astype(float).tolist())
    assert gradient is not None
    np.testing.assert_allclose(gradient, 2.0 * x0, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("test_point", "description"),
    [
        (np.array([0.0, 0.0, 0.0], dtype=np.float32), "origin"),
        (np.array([1.0, 1.0, 1.0], dtype=np.float32), "all ones"),
        (np.array([-1.0, -1.0, -1.0], dtype=np.float32), "all negative ones"),
        (np.array([10.0, -10.0, 0.0], dtype=np.float32), "large values"),
    ],
)
def test_gradient_at_special_points(test_point, description):
    """Test gradient computation at special points: {description}"""
    # Arrange
    expected_gradient = 2.0 * test_point

    # Compute gradient
    computed_gradient = quadratic_problem().evaluate_gradient(
        test_point.astype(float).tolist()
    )

    # Assert
    assert computed_gradient is not None, f"Gradient at {description} returned None"
    np.testing.assert_allclose(
        computed_gradient,
        expected_gradient,
        rtol=1e-5,
        atol=1e-6,
        err_msg=f"Gradient incorrect at {description}: {test_point}",
    )
