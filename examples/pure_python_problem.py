import chronopt as chron


# Example function
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# Simple API
builder = (
    chron.PythonBuilder()
    .add_callable(rosenbrock)
    .add_parameter("x")
    .set_optimiser(chron.NelderMead().with_max_iter(1000))
)
problem = builder.build()
mle = problem.optimize(x0=[10.0, 10.0])

print(mle)
print(f"Optimal x: {mle.x}")
print(f"Optimal value: {mle.fun}")
