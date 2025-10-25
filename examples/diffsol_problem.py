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
t_span = np.linspace(0, 1, 10)
# Simple logistic growth for testing
data = 0.1 * np.exp(t_span) / (1 + 0.1 * (np.exp(t_span) - 1))

params = {"r": 1.0, "k": 1.0}

# Simple API
builder = (
    chron.DiffsolBuilder()
    .add_diffsl(ds)
    .add_data(data)
    .with_rtol(1e-6)
    .add_params(params)
)
problem = builder.build()

# Optimize to find MAP estimate
map_result = problem.optimize()
print(f"MAP result: {map_result}")

# For now, just print the optimization result since Hamiltonian sampler is not implemented yet
print(f"Optimal parameters: {map_result.x}")
print(f"Optimal cost: {map_result.fun}")
print(f"Optimization success: {map_result.success}")
print(f"Iterations: {map_result.nit}")

# Note: Hamiltonian sampler will be implemented in Phase 3
print("\nNote: Hamiltonian sampler will be implemented in Phase 3")
