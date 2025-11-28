import chronopt as chron
import numpy as np


def quadratic(x):
    x0 = x[0] - 1.5
    x1 = x[1] + 0.5
    return np.asarray([x0**2 + x1**2], dtype=float)


def quadratic_grad(x):
    x0 = x[0] - 1.5
    x1 = x[1] + 0.5
    return np.asarray([2.0 * x0, 2.0 * x1], dtype=float)


def build_quadratic_problem_with_gradient():
    return (
        chron.ScalarBuilder()
        .with_callable(quadratic)
        .with_gradient(quadratic_grad)
        .with_parameter("x", 1.0)
        .with_parameter("y", 1.0)
        .build()
    )


def test_adam_direct_run_minimises_quadratic():
    problem = build_quadratic_problem_with_gradient()

    optimiser = chron.Adam().with_step_size(0.1).with_max_iter(500).with_threshold(1e-8)

    result = optimiser.run(problem, [5.0, -4.0])

    assert result.success
    assert result.fun < 1e-6
    assert np.allclose(result.x, np.array([1.5, -0.5]), atol=1e-2)


def test_python_builder_optimise_with_adam_default():
    builder = (
        chron.ScalarBuilder()
        .with_callable(quadratic)
        .with_gradient(quadratic_grad)
        .with_parameter("x", 1.0)
        .with_parameter("y", 1.0)
    )

    optimiser = chron.Adam().with_step_size(0.1).with_max_iter(400).with_threshold(1e-8)

    builder.with_optimiser(optimiser)
    problem = builder.build()

    result = problem.optimize(initial=[3.0, -3.0])

    assert result.success
    assert result.fun < 1e-5
    assert np.allclose(result.x, np.array([1.5, -0.5]), atol=1e-2)


def test_adam_requires_gradient():
    builder = (
        chron.ScalarBuilder()
        .with_callable(quadratic)
        .with_parameter("x", 0.0)
        .with_parameter("y", 0.0)
    )

    problem = builder.build()

    optimiser = chron.Adam().with_max_iter(10)
    result = optimiser.run(problem, [0.0, 0.0])

    assert not result.success
    assert "requires an available gradient" in result.message
