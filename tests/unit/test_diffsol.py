import copy

import diffid
import numpy as np
import pytest


def test_diffsol_builder():
    """Test basic Diffsol builder functionality"""
    # Example diffsol ODE (logistic growth)
    ds = """
in_i { r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

    # Generate some test data
    t_span = np.linspace(0, 1, 100)
    # Simple exponential growth for testing
    data = 0.1 * np.exp(t_span)
    stacked_data = np.column_stack((t_span, data))

    # Build the problem
    builder = (
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_tolerances(rtol=1e-6, atol=1e-6)
        .with_parameter("r", 1.0, None)
        .with_parameter("k", 1.0, None)
        .with_cost(diffid.SSE())
        .with_cost(diffid.RMSE())
    )

    problem = builder.build()

    # Test that we can evaluate the problem
    x0 = [1.0, 1.0]  # r, k parameters
    cost = problem.evaluate(x0)

    # Cost should be finite
    assert np.isfinite(cost), f"Cost should be finite, got {cost}"
    assert cost >= 0, f"Cost should be non-negative, got {cost}"

    # Test that we can optimise the problem
    optimiser = (
        diffid.NelderMead().with_max_iter(500).with_threshold(1e-7).with_patience(10)
    )
    result = optimiser.run(problem, x0)
    assert result.success
    assert result.value < 1e-5


def test_diffsol_builder_remove_methods():
    ds = """
in_i { a = 1 }
u_i { y = 0.0 }
F_i { a * y }
"""

    t_span = np.linspace(0, 1, 5)
    data = t_span**2
    stacked_data = np.column_stack((t_span, data))

    metric = diffid.RMSE()

    builder = (
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_parameter("a", 1.0, None)
        .with_cost(metric)
    )

    # Initial build
    builder_copy = copy.deepcopy(builder)
    problem_1 = builder_copy.build()

    # Change data
    builder = builder.remove_data()
    builder = builder.with_data(np.column_stack((t_span, t_span**3)))
    problem_2 = builder.build()

    # Change t_span along with data
    builder = builder.remove_data()
    new_t_span = t_span * 2
    builder = builder.with_data(np.column_stack((new_t_span, data)))
    problem_3 = builder.build()

    # Change params
    builder = builder.clear_parameters()
    builder = builder.with_parameter("a", 2.0, None)
    problem_4 = builder.build()

    # Change cost
    builder = builder.remove_cost()
    builder = builder.with_cost(diffid.SSE())
    problem_5 = builder.build()

    # Check that problems are different
    assert problem_1 != problem_2
    assert problem_2 != problem_3
    assert problem_3 != problem_4
    assert problem_4 != problem_5

    assert problem_1.evaluate([1.0]) != problem_5.evaluate([1.0])
    assert problem_1.evaluate([1.0]) != problem_2.evaluate([1.0])


def test_problem_optimise_defaults_to_builder_params():
    ds = """
in_i { a = 1 }
u_i { y = 0.1 }
F_i { a * y }
"""

    t_span = np.linspace(0, 1, 6)
    true_param = 2.5
    data = 0.1 * np.exp(true_param * t_span)
    stacked_data = np.column_stack((t_span, data))

    problem = (
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_tolerances(rtol=1e-4, atol=1e-4)
        .with_data(stacked_data)
        .with_parameter("a", true_param)
        .build()
    )

    optimiser = diffid.NelderMead().with_max_iter(0)

    result = problem.optimise(optimiser=optimiser)

    assert problem.config()["rtol"] == 1e-4
    assert pytest.approx(true_param, rel=1e-12, abs=1e-12) == result.x[0]
    assert result.iterations == 0
    assert pytest.approx(true_param, rel=1e-12, abs=1e-12) == result.final_simplex[0][0]


@pytest.mark.parametrize("variance", [0.5, 2.0])
def test_diffsol_cost_metrics(variance: float) -> None:
    """Ensure selectable cost metrics produce consistent values."""

    ds = """
in_i { r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

    time_points = np.linspace(0, 1, 20)
    data = 0.1 * np.exp(time_points)
    stacked_data = np.column_stack((time_points, data))

    def build_problem(cost_metric=None):
        builder = (
            diffid.DiffsolBuilder()
            .with_diffsl(ds)
            .with_data(stacked_data)
            .with_tolerances(rtol=1e-6, atol=1e-6)
            .with_parameter("r", 2.0)
            .with_parameter("k", 4.0)
        )
        if cost_metric is not None:
            builder = builder.with_cost(cost_metric)
        return builder.build()

    sse_problem = build_problem()
    sse_problem_explicit = build_problem(diffid.SSE())
    rmse_problem = build_problem(diffid.RMSE())
    gaussian_problem = build_problem(diffid.GaussianNLL(variance))

    test_params = [0.8, 1.2]
    sse_cost = sse_problem.evaluate(test_params)
    assert np.isfinite(sse_cost)
    assert sse_cost >= 0

    sse_cost_explicit = sse_problem_explicit.evaluate(test_params)
    assert pytest.approx(sse_cost, rel=1e-8, abs=1e-10) == sse_cost_explicit

    rmse_cost = rmse_problem.evaluate(test_params)
    expected_rmse = np.sqrt(sse_cost / data.size)
    assert pytest.approx(expected_rmse, rel=1e-6, abs=1e-9) == rmse_cost

    gaussian_cost = gaussian_problem.evaluate(test_params)
    expected_gaussian = (
        0.5 * data.size * np.log(2.0 * np.pi * variance) + 0.5 * sse_cost / variance
    )
    assert pytest.approx(expected_gaussian, rel=1e-6, abs=1e-9) == gaussian_cost

    with pytest.raises(ValueError, match="variance must be positive"):
        diffid.GaussianNLL(0.0)


def test_diffsol_bicycle_model_neldermead_recovers_wheelbase() -> None:
    ds = """
in_i { L = 2.5 }
v { 5.0 } delta { 0.05 }
u_i {
    y = 0.0,
    psi = 0.0,
}
F_i {
    v * psi,
    v / L * delta,
}
"""

    true_L = 2.5
    v = 5.0
    delta = 0.05

    t_span = np.linspace(0.0, 2.0, 51)
    psi_true = (v / true_L) * delta * t_span
    y_true = 0.5 * v * (v / true_L) * delta * t_span**2

    stacked_data = np.column_stack((t_span, y_true, psi_true))

    builder = (
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_tolerances(rtol=1e-6, atol=1e-8)
        .with_parameter("L", 4.0)
    )

    problem = builder.build()

    optimiser = (
        diffid.NelderMead()
        .with_max_iter(500)
        .with_threshold(1e-10)
        .with_position_tolerance(1e-8)
    )

    result = problem.optimise(optimiser=optimiser)

    assert result.success
    assert pytest.approx(true_L, rel=1e-2, abs=1e-2) == result.x[0]
    assert result.value < 1e-6
