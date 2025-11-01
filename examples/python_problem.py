import chronopt as chron
import numpy as np


# Example function
def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.array([value], dtype=float)


# Simple API
builder = (
    chron.PythonBuilder()
    .add_callable(rosenbrock)
    .add_parameter("x")
    .add_parameter("y")
    .set_optimiser(chron.NelderMead().with_max_iter(1000))
)
problem = builder.build()
mle = problem.optimize(initial=[10.0, 10.0])

print(mle)
print(f"Optimal x: {mle.x}")
print(f"Optimal value: {mle.fun}")
