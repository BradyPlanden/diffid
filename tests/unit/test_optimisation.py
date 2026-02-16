import diffid
import numpy as np
import pytest


def test_builder_exposes_config_and_parameters():
    builder = (
        diffid.ScalarBuilder()
        .with_objective(lambda x: np.asarray([float(x[0]) ** 2]))
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
        diffid.ScalarBuilder()
        .with_objective(rosenbrock)
        .with_parameter("x", 1.2, None)
        .with_parameter("y", -1.2, None)
    )
    problem = builder.build()

    # Create the optimisation
    optimiser = (
        diffid.NelderMead().with_max_iter(500).with_threshold(1e-6).with_step_size(0.15)
    )
    results = optimiser.run(problem, [1.5, -1.5])

    # Validation metrics
    assert results.success
    assert np.allclose(results.x, np.ones(2), atol=1e-3)
    assert results.value < 1e-6


def test_python_builder_bounds_respected():
    builder = (
        diffid.ScalarBuilder()
        .with_objective(bounded_quadratic)
        .with_parameter("x", 0.0, bounds=(0.0, 1.0))
        .with_parameter("y", 0.0, bounds=(0.0, 2.0))
    )
    problem = builder.build()

    optimiser = diffid.NelderMead().with_max_iter(200).with_threshold(1e-8)
    results = optimiser.run(problem, [0.5, 1.0])

    assert results.success
    assert 0.0 <= results.x[0] <= 1.0
    assert 0.0 <= results.x[1] <= 2.0
    assert np.allclose(results.x, np.array([1.0, 2.0]), atol=1e-2)


def test_scalar_builder_objective_exception_raises_evaluation_error():
    def failing_objective(_x):
        raise RuntimeError("objective boom")

    problem = (
        diffid.ScalarBuilder()
        .with_objective(failing_objective)
        .with_parameter("x", 1.0)
        .build()
    )

    with pytest.raises(diffid.errors.EvaluationError, match="objective boom"):
        problem.evaluate([1.0])


def test_scalar_builder_gradient_exception_raises_evaluation_error():
    def objective(x):
        return x[0] ** 2

    def failing_gradient(_x):
        raise RuntimeError("gradient boom")

    problem = (
        diffid.ScalarBuilder()
        .with_objective(objective)
        .with_gradient(failing_gradient)
        .with_parameter("x", 1.0)
        .build()
    )

    with pytest.raises(diffid.errors.EvaluationError, match="gradient boom"):
        problem.evaluate_gradient([1.0])


def test_scalar_builder_optimise_surfaces_objective_failure_details():
    def failing_objective(_x):
        raise RuntimeError("optimise objective boom")

    problem = (
        diffid.ScalarBuilder()
        .with_objective(failing_objective)
        .with_parameter("x", 1.0)
        .build()
    )

    result = diffid.NelderMead().with_max_iter(5).run(problem, [1.0])

    assert not result.success
    assert "optimise objective boom" in result.message
    assert "non-finite objective value" not in result.message


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_scalar_builder_rejects_non_finite_initial_values_without_bounds(value):
    builder = diffid.ScalarBuilder().with_objective(lambda x: x[0] ** 2)

    with pytest.raises(ValueError, match="must be finite"):
        builder.with_parameter("x", value)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_scalar_builder_rejects_non_finite_initial_values_with_bounds(value):
    builder = diffid.ScalarBuilder().with_objective(lambda x: x[0] ** 2)

    with pytest.raises(ValueError, match="must be finite"):
        builder.with_parameter("x", value, bounds=(-1.0, 1.0))
