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

    # Build the problem
    builder = (
        chron.DiffsolBuilder()
        .add_diffsl(ds)
        .add_data(data)
        .add_config({"rtol": 1e-6})
        .add_params({"r": 1.0, "k": 1.0})
    )

    problem = builder.build()

    # Test that we can optimise the problem
    result = problem.optimize()
    assert result.success
    assert result.fun < 1e-5
