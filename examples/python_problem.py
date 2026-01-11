import chronopt as chron
import numpy as np


# Example function
def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.array([value], dtype=float)


# Simple API
builder = (
    chron.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", 1.0)
    .with_parameter("y", 1.0)
    .with_optimiser(chron.NelderMead().with_max_iter(1000))
)
problem = builder.build()
result = problem.optimise(initial=[10.0, 10.0])

print(result)
print(f"Optimal x: {result.x}")
print(f"Optimal function val: {result.value}")
