import diffid
import numpy as np

# Example diffsol ODE (logistic growth)
ds = """
in_i { r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

# Generate some test data (logistic growth)
t_span = np.linspace(0, 1, 100)
data = 0.1 * np.exp(t_span) / (1 + 0.1 * (np.exp(t_span) - 1))
stacked_data = np.column_stack((t_span, data))


# Create an optimiser
optimiser = diffid.CMAES().with_max_iter(1000).with_threshold(1e-12)

# Simple API
builder = (
    diffid.DiffsolBuilder()
    .with_diffsl(ds)
    .with_data(stacked_data)
    .with_tolerances(1e-6, 1e-8)
    .with_parameter("r", 100)
    .with_parameter("k", 100)
    .with_parallel(True)
    .with_optimiser(optimiser)  # Override default optimiser
)
problem = builder.build()

# Optimise
results = problem.optimise()

print(f"result: {results}")

# For now, just print the optimisation result since Hamiltonian sampler is not implemented yet
print(f"Optimal parameters: {results.x}")
print(f"Optimal cost: {results.value}")
print(f"Optimisation success: {results.success}")
print(f"Iterations: {results.iterations}")
print(f"Optimisation time: {results.time}")
