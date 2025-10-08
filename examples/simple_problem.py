import chronopt as chron


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


builder = chron.builder.thing().add_callable(rosenbrock)
problem = builder.build()
optim = chron.NelderMead(problem)
results = optim.run()

print(results)  # OptimizationResults(x=[...], fun=..., nit=..., success=True)
print(f"Optimal x: {results.x}")
print(f"Optimal value: {results.fun}")
