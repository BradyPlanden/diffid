import diffid
import numpy as np


def _test_optimisation_api():
    """Test basic Diffsol builder functionality"""
    # Example diffsol ODE (logistic growth)
    ds = """
in_i { r = 1, k = 1 }
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
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_config({"rtol": 1e-6})
        .with_parameter("r", 1.0)
        .with_parameter("k", 1.0)
    )

    problem = builder.build()

    # Test that we can optimise the problem
    result = problem.optimise()
    assert result.success
    assert result.value < 1e-5


def test_diffsol_builder_allows_multiple_builds():
    ds = """
in_i { r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

    t_span = np.linspace(0, 1, 10)
    data = 0.1 * np.exp(t_span)
    stacked_data = np.column_stack((t_span, data))

    builder = (
        diffid.DiffsolBuilder()
        .with_diffsl(ds)
        .with_data(stacked_data)
        .with_parameter("r", 1.0)
        .with_parameter("k", 1.0)
    )

    problem_1 = builder.build()
    problem_2 = builder.build()

    assert problem_1.dimension() == problem_2.dimension() == 2

    result = problem_1.optimise()
    assert result.success


def test_scalar_builder_allows_multiple_builds():
    """ScalarBuilder should support calling build() multiple times"""

    def rosenbrock(x):
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

    builder = (
        diffid.ScalarBuilder()
        .with_objective(rosenbrock)
        .with_parameter("x", 1.0)
        .with_parameter("y", 1.0)
    )

    problem_1 = builder.build()
    problem_2 = builder.build()

    assert problem_1.dimension() == problem_2.dimension() == 2


def test_vector_builder_allows_multiple_builds():
    """VectorBuilder should support calling build() multiple times"""
    t_span = np.linspace(0, 2, 50)
    data = 0.5 * np.exp(1.5 * t_span)

    def exponential_model(params):
        rate, y0 = params
        return y0 * np.exp(rate * t_span)

    builder = (
        diffid.VectorBuilder()
        .with_objective(exponential_model)
        .with_data(data)
        .with_parameter("rate", 1.0)
        .with_parameter("y0", 1.0)
    )

    problem_1 = builder.build()
    problem_2 = builder.build()

    assert problem_1.dimension() == problem_2.dimension() == 2


def test_all_builders_support_copy():
    """All builders should support copy.copy() and copy.deepcopy()"""
    import copy

    # ScalarBuilder
    scalar_builder = diffid.ScalarBuilder().with_objective(lambda x: x[0] ** 2)
    scalar_copy = copy.copy(scalar_builder)
    copy.deepcopy(scalar_builder)  # Test deepcopy

    # DiffsolBuilder
    diffsol_builder = diffid.DiffsolBuilder().with_diffsl("in { a }")
    diffsol_copy = copy.copy(diffsol_builder)
    copy.deepcopy(diffsol_builder)  # Test deepcopy

    # VectorBuilder
    vector_builder = diffid.VectorBuilder().with_objective(lambda x: [x[0]])
    vector_copy = copy.copy(vector_builder)
    copy.deepcopy(vector_builder)  # Test deepcopy

    # All should succeed without error
    assert scalar_copy is not scalar_builder
    assert diffsol_copy is not diffsol_builder
    assert vector_copy is not vector_builder
