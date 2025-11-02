import chronopt as chron
import numpy as np


# Build an optimisation problem
def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value], dtype=float)


def test_python_builder_rosenbrock():
    builder = (
        chron.PythonBuilder()
        .add_callable(rosenbrock)
        .add_parameter("x")
        .add_parameter("y")
    )
    problem = builder.build()

    # Create the optimisation
    optimiser = chron.NelderMead().with_max_iter(500).with_threshold(1e-6)
    results = optimiser.run(problem, [1.5, -1.5])

    # Validation metrics
    assert results.success
    assert np.allclose(results.x, np.ones(2), atol=1e-3)
    assert results.fun < 1e-6
