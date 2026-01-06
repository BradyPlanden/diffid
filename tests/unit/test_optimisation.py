import chronopt as chron
import numpy as np


def test_builder_exposes_config_and_parameters():
    builder = (
        chron.ScalarBuilder()
        .with_callable(lambda x: np.asarray([float(x[0]) ** 2]))
        .with_parameter("x", 3.5, bounds=(0.0, 10.0))
    )

    problem = builder.build()

    params = problem.parameters()
    assert params == [("x", 3.5, (0.0, 10.0))]

    assert problem.default_parameters() == [3.5]


# Build an optimisation problem
def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value], dtype=float)


def bounded_quadratic(x):
    value = (x[0] - 2.0) ** 2 + (x[1] - 3.0) ** 2
    return np.asarray([value], dtype=float)


def test_python_builder_rosenbrock():
    builder = (
        chron.ScalarBuilder()
        .with_callable(rosenbrock)
        .with_parameter("x", 1.2, None)
        .with_parameter("y", -1.2, None)
    )
    problem = builder.build()

    # Create the optimisation
    optimiser = chron.NelderMead().with_max_iter(500).with_threshold(1e-6).with_step_size(0.15)
    results = optimiser.run(problem, [1.5, -1.5])

    # Validation metrics
    assert results.success
    assert np.allclose(results.x, np.ones(2), atol=1e-3)
    assert results.value < 1e-6


def test_python_builder_bounds_respected():
    builder = (
        chron.ScalarBuilder()
        .with_callable(bounded_quadratic)
        .with_parameter("x", 0.0, bounds=(0.0, 1.0))
        .with_parameter("y", 0.0, bounds=(0.0, 2.0))
    )
    problem = builder.build()

    optimiser = chron.NelderMead().with_max_iter(200).with_threshold(1e-8)
    results = optimiser.run(problem, [0.5, 1.0])

    assert results.success
    assert 0.0 <= results.x[0] <= 1.0
    assert 0.0 <= results.x[1] <= 2.0
    assert np.allclose(results.x, np.array([1.0, 2.0]), atol=1e-2)
