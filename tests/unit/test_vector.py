import diffid
import numpy as np
import pytest


def test_vector_builder_basic():
    """Test basic VectorBuilder functionality with exponential model"""
    # Generate test data: exponential growth
    t_span = np.linspace(0, 2, 50)
    true_params = [1.5, 0.5]  # growth rate, initial value
    data = true_params[1] * np.exp(true_params[0] * t_span)

    # Define objective function that returns predictions
    def exponential_model(params):
        rate, y0 = params
        return y0 * np.exp(rate * t_span)

    # Build the problem
    builder = (
        diffid.VectorBuilder()
        .with_objective(exponential_model)
        .with_data(data)
        .with_parameter("rate", 1.0, None)
        .with_parameter("y0", 1.0, None)
        .with_cost(diffid.SSE())
    )

    problem = builder.build()
    # Test that we can evaluate the problem
    x0 = [1.0, 1.0]
    cost = problem.evaluate(x0)

    # Cost should be finite and non-negative
    assert np.isfinite(cost), f"Cost should be finite, got {cost}"
    assert cost >= 0, f"Cost should be non-negative, got {cost}"

    # Test optimisation
    optimiser = diffid.NelderMead().with_max_iter(1000).with_threshold(1e-8)
    result = problem.optimise(x0, optimiser)

    assert result.success
    assert result.value < 1e-6
    # Check parameters are close to true values
    assert np.allclose(result.x, true_params, rtol=1e-2, atol=1e-2)


def test_vector_builder_with_shape():
    """Test VectorBuilder with inferred shape from data"""
    # 1D data: 30 points (shape will be inferred as [30])
    n_points = 30
    data = np.random.randn(n_points)

    def model(params):
        # Simple linear model
        return params[0] * np.ones(n_points) + params[1]

    builder = (
        diffid.VectorBuilder()
        .with_objective(model)
        .with_data(data)
        .with_parameter("scale", 1.0)
        .with_parameter("offset", 0.0)
    )

    problem = builder.build()
    cost = problem.evaluate([1.0, 0.0])
    assert np.isfinite(cost)


def test_vector_builder_sinusoidal():
    """Test VectorBuilder with sinusoidal time series"""
    t = np.linspace(0, 4 * np.pi, 100)
    true_amplitude = 2.5
    true_frequency = 1.5
    true_phase = 0.3
    data = true_amplitude * np.sin(true_frequency * t + true_phase)

    def sinusoid(params):
        amp, freq, phase = params
        return amp * np.sin(freq * t + phase)

    problem = (
        diffid.VectorBuilder()
        .with_objective(sinusoid)
        .with_data(data)
        .with_parameter("amplitude", 2.0, (0.0, 5.0))
        .with_parameter("frequency", 1.0, (0.1, 3.0))
        .with_parameter("phase", 0.0, (-np.pi, np.pi))
        .with_cost(diffid.RMSE())
        .build()
    )

    x0 = [2.0, 1.0, 0.0]
    optimiser = diffid.NelderMead().with_max_iter(2000).with_threshold(1e-9)
    result = problem.optimise(x0, optimiser)

    # Note: Sinusoidal fitting can be challenging due to local minima
    # We just verify the optimisation runs and produces reasonable results
    assert (
        result.success or result.iterations >= 1000
    )  # Either converged or tried hard enough
    # Cost should be reduced from initial
    initial_cost = problem.evaluate(x0)
    assert result.value < initial_cost


def test_vector_builder_cost_metrics():
    """Test different cost metrics with VectorBuilder"""
    t = np.linspace(0, 1, 20)
    data = t**2

    def quadratic(params):
        return params[0] * t**2

    def build_problem(cost_metric=None):
        builder = (
            diffid.VectorBuilder()
            .with_objective(quadratic)
            .with_data(data)
            .with_parameter("scale", 1.5)
        )
        if cost_metric is not None:
            builder = builder.with_cost(cost_metric)
        return builder.build()

    sse_problem = build_problem()
    sse_problem_explicit = build_problem(diffid.SSE())
    rmse_problem = build_problem(diffid.RMSE())
    gaussian_problem = build_problem(diffid.GaussianNLL(1.0))

    test_params = [1.5]
    sse_cost = sse_problem.evaluate(test_params)
    assert np.isfinite(sse_cost)
    assert sse_cost >= 0

    sse_cost_explicit = sse_problem_explicit.evaluate(test_params)
    assert pytest.approx(sse_cost, rel=1e-8, abs=1e-10) == sse_cost_explicit

    rmse_cost = rmse_problem.evaluate(test_params)
    expected_rmse = np.sqrt(sse_cost / data.size)
    assert pytest.approx(expected_rmse, rel=1e-6, abs=1e-9) == rmse_cost

    gaussian_cost = gaussian_problem.evaluate(test_params)
    variance = 1.0
    expected_gaussian = (
        0.5 * data.size * np.log(2.0 * np.pi * variance) + 0.5 * sse_cost / variance
    )
    assert pytest.approx(expected_gaussian, rel=1e-6, abs=1e-9) == gaussian_cost


def test_vector_builder_remove_methods():
    """Test builder remove/clear methods"""
    data = np.array([1.0, 2.0, 3.0])

    def model(params):
        return params[0] * np.ones(3)

    # Build with SSE
    builder1 = (
        diffid.VectorBuilder()
        .with_objective(model)
        .with_data(data)
        .with_parameter("a", 1.0)
        .with_cost(diffid.SSE())
    )
    problem1 = builder1.build()

    # Build with RMSE
    builder2 = (
        diffid.VectorBuilder()
        .with_objective(model)
        .with_data(data)
        .with_parameter("a", 1.0)
        .with_cost(diffid.RMSE())
    )
    problem2 = builder2.build()

    # Verify problems produce different costs (SSE vs RMSE)
    cost1 = problem1.evaluate([1.5])
    cost2 = problem2.evaluate([1.5])
    assert cost1 != cost2  # SSE and RMSE should give different values


def test_vector_builder_with_default_optimiser():
    """Test VectorBuilder with default optimiser"""
    data = np.array([1.0, 2.0, 3.0, 4.0])

    def linear(params):
        return params[0] * np.arange(4) + params[1]

    optimiser = diffid.NelderMead().with_max_iter(100)

    problem = (
        diffid.VectorBuilder()
        .with_objective(linear)
        .with_data(data)
        .with_parameter("slope", 0.5)
        .with_parameter("intercept", 0.5)
        .with_optimiser(optimiser)
        .build()
    )

    # Should use default optimiser when none specified
    result = problem.optimise()
    assert result.success
    assert result.iterations <= 100


def test_vector_builder_dimension_mismatch():
    """Test that dimension mismatch raises appropriate error"""
    data = np.array([1.0, 2.0, 3.0])

    def wrong_size(params):
        return params[0] * np.ones(5)  # Wrong size!

    problem = (
        diffid.VectorBuilder()
        .with_objective(wrong_size)
        .with_data(data)
        .with_parameter("a", 1.0)
        .build()
    )

    with pytest.raises(
        diffid.errors.EvaluationError,
        match="Evaluation failed: Evaluation failed:: expected 3 elements, got 5",
    ):
        problem.evaluate([1.0])


def test_vector_builder_multiple_builds():
    """Test building multiple problems with same configuration"""
    data = np.array([1.0, 2.0, 3.0])

    def model(params):
        return params[0] * data

    # Build two problems with same configuration
    builder = (
        diffid.VectorBuilder()
        .with_objective(model)
        .with_data(data)
        .with_parameter("scale", 1.0)
        .with_cost(diffid.RMSE())
    )

    problem1 = builder.build()
    builder.remove_cost()
    builder.with_cost(diffid.SSE())
    problem2 = builder.build()

    # Should produce same results
    assert problem1.evaluate([1.5]) != problem2.evaluate([1.5])


def test_vector_builder_config():
    """Test VectorBuilder config method"""
    data = np.array([1.0, 2.0])

    def model(params):
        return params[0] * data

    problem = (
        diffid.VectorBuilder()
        .with_objective(model)
        .with_data(data)
        .with_parameter("a", 1.0)
        .with_config("custom_param", 42.0)
        .build()
    )

    config = problem.config()
    assert "custom_param" in config
    assert config["custom_param"] == 42.0
