import chronopt as chron
import numpy as np


def test_diffsol_builder():
    """Test basic Diffsol builder functionality"""
    # Example diffsol ODE (logistic growth)
    ds = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

    # Generate some test data
    t_span = np.linspace(0, 1, 10)
    # Simple exponential growth for testing
    data = 0.1 * np.exp(t_span)

    # Build the problem
    builder = (
        chron.DiffsolBuilder()
        .add_diffsl(ds)
        .add_data(data)
        .add_config({"rtol": 1e-6})
        .add_params({"r": 1.0, "k": 1.0})
    )

    problem = builder.build()

    # Test that we can evaluate the problem
    x0 = [1.0, 1.0]  # r, k parameters
    cost = problem.evaluate(x0)

    # Cost should be finite
    assert np.isfinite(cost), f"Cost should be finite, got {cost}"
    assert cost >= 0, f"Cost should be non-negative, got {cost}"

    # Test that we can optimise the problem
    optimiser = chron.NelderMead().with_max_iter(500).with_threshold(1e-6)
    result = optimiser.run(problem, x0)
    assert result.success
    assert result.fun < 1e-5
