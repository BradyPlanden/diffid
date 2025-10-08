import chronopt as chron


# Build an optimisation problem
def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


builder = chron.builders.Problem().add_callable(rosenbrock)
problem = builder.build()

# Create the optimisation
optim = chron.optim.CMAES()
optim.set_initial_scale(1.0)
optim.set_time_allowance(60.0)
optim.set_convergence_threshold(1e-5)
optim.set_patience(10)

results = optim.run()

# Validation metrics
# Some may require additional arguments, or further computation
hessian = results.hessian()
evidence = results.evidence()
sensitivities = results.sensitivities()
sloppy_parameters = results.sloppy()


builder = package.builder.thing().add_callable(obj)
problem = builder.build()

optim = package.NelderMead(problem)

results = optim.run()
