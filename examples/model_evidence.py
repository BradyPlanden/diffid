import chronopt as chron


# Example function
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


# Simple API
builder = (
    chron.Builder()
    .add_callable(rosenbrock)
    .add_parameter("x", prior=chron.Normal(0, 1))
    .set_optimiser(chron.NelderMead().with_max_iter(1000))
)
problem = builder.build()

sampler = chron.DynamicNestedSampler(problem)
log_z = sampler.run()

print(log_z)
