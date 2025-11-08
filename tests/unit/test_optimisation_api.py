import chronopt as chron
import numpy as np


def _test_optimisation_api():
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
    stacked_data = np.column_stack((t_span, data))

    # Build the problem
    builder = (
        chron.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_config({"rtol": 1e-6})
        .with_parameter("r", 1.0)
        .with_parameter("k", 1.0)
    )

    problem = builder.build()

    # Test that we can optimise the problem
    result = problem.optimize()
    assert result.success
    assert result.fun < 1e-5


def test_diffsol_builder_allows_multiple_builds():
    ds = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

    t_span = np.linspace(0, 1, 10)
    data = 0.1 * np.exp(t_span)
    stacked_data = np.column_stack((t_span, data))

    builder = (
        chron.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_parameter("r", 1.0)
        .with_parameter("k", 1.0)
    )

    problem_1 = builder.build()
    problem_2 = builder.build()

    assert problem_1.dimension() == problem_2.dimension() == 2

    result = problem_1.optimize()
    assert result.success
