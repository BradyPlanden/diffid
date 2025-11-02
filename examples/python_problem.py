import chronopt as chron
import numpy as np

dsl = """
state x
param k
dx/dt = -k * x
"""

t = np.linspace(0.0, 5.0, 51)
observations = np.exp(-1.3 * t)
data = np.column_stack((t, observations))

builder = (
    chron.DiffsolBuilder()
    .add_diffsl(dsl)
    .add_data(data)
    .add_params({"k": 1.0})
    .with_backend("dense")
)
problem = builder.build()

optimiser = chron.CMAES().with_max_iter(1000)
result = optimiser.run(problem, [0.5])

print(result.x)
