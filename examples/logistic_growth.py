import chronopt as chron
import numpy as np

# Example diffsol ODE (logistic growth)
ds = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

# Generate some test data (logistic growth)
t_span = np.linspace(0, 1, 100)
data = 0.1 * np.exp(t_span) / (1 + 0.1 * (np.exp(t_span) - 1))
stacked_data = np.column_stack((t_span, data))


# Create an optimiser
optimiser = chron.CMAES().with_max_iter(1000).with_threshold(1e-12).with_sigma0(1.0)

# Simple API
builder = (
    chron.DiffsolBuilder()
    .with_diffsl(ds)
    .with_data(stacked_data)
    .with_rtol(1e-6)
    .with_atol(1e-8)
    .with_parameter("r", 1000)
    .with_parameter("k", 1000)
    .with_parallel(True)
    .with_optimiser(optimiser)  # Override default optimiser
)
problem = builder.build()

# Optimize
results = problem.optimize()

print(f"result: {results}")

# For now, just print the optimization result since Hamiltonian sampler is not implemented yet
print(f"Optimal parameters: {results.x}")
print(f"Optimal cost: {results.fun}")
print(f"Optimization success: {results.success}")
print(f"Iterations: {results.nit}")
print(f"Optimisation time: {results.time}")
