import chronopt as chron
import numpy as np
import pytest


def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value], dtype=float)


def build_rosenbrock_problem():
    return (
        chron.PythonBuilder()
        .add_callable(rosenbrock)
        .add_parameter("x")
        .add_parameter("y")
        .build()
    )


def test_cmaes_direct_run_minimises_rosenbrock():
    problem = build_rosenbrock_problem()

    optimiser = (
        chron.CMAES()
        .with_max_iter(400)
        .with_threshold(1e-8)
        .with_sigma0(0.6)
        .with_seed(42)
    )

    result = optimiser.run(problem, [5.0, -4.0])

    assert result.success
    assert result.fun < 1e-6
    assert np.allclose(result.x, np.ones(2), atol=1e-2)


def test_python_builder_optimise_with_cmaes_default():
    builder = (
        chron.PythonBuilder()
        .add_callable(rosenbrock)
        .add_parameter("x")
        .add_parameter("y")
    )

    optimiser = (
        chron.CMAES()
        .with_max_iter(300)
        .with_threshold(1e-8)
        .with_sigma0(0.5)
        .with_seed(7)
    )

    builder.set_optimiser(optimiser)
    problem = builder.build()

    result = problem.optimize(initial=[3.0, -3.0])

    assert result.success
    assert result.fun < 1e-5
    assert np.allclose(result.x, np.ones(2), atol=1e-2)


def test_set_optimiser_rejects_unknown_type():
    builder = chron.PythonBuilder()

    with pytest.raises(TypeError):
        builder.set_optimiser(object())


def test_cmaes_result_covariance_available():
    problem = build_rosenbrock_problem()

    optimiser = chron.CMAES().with_max_iter(50).with_seed(123)

    result = optimiser.run(problem, [1.5, -1.5])

    covariance = result.covariance

    assert covariance is not None
    assert len(covariance) == 2
    assert len(covariance[0]) == 2
