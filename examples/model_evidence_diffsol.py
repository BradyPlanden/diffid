"""Model evidence example using the dynamic nested sampler."""

from __future__ import annotations

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

builder = (
    chron.DiffsolBuilder()
    .with_diffsl(ds)
    .with_data(stacked_data)
    .with_parameter("r", initial_value=1.2)
    .with_parameter("k", initial_value=1.4)
    .with_parallel(True)
)
problem = builder.build()

optimised = problem.optimise()

sampler = chron.DynamicNestedSampler().with_live_points(256).with_seed(1234)
samples = sampler.run(problem, initial=optimised.x)

print("time       :", samples.time)
print("log(Z)      :", samples.log_evidence)
print("information :", samples.information)
print("evaluations :", samples.draws)
