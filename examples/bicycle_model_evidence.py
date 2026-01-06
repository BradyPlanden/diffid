from __future__ import annotations

import chronopt as chron
import numpy as np

TRUE_L = 2.5  # wheelbase
V = 5.0  # velocity
DELTA = 0.05  # steer angle

dsl = """
in = [L]
L { 2.5 } v { 5.0 } delta { 0.05 }
u_i {
    x = 0.0,
    y = 0.0,
    theta = 0.0,
}
F_i {
    v * cos(theta),
    v * sin(theta),
    v / L * sin(delta)/cos(delta),
}
"""

# Synthetic data generation
t_span = np.linspace(0.0, 2.0, 201)
omega = (V / TRUE_L) * np.tan(DELTA)
psi_true = omega * t_span
x_true = (TRUE_L / np.tan(DELTA)) * np.sin(omega * t_span)
y_true = (TRUE_L / np.tan(DELTA)) * (1.0 - np.cos(omega * t_span))

rng = np.random.default_rng(123)
x_obs = x_true + rng.normal(scale=0.01, size=x_true.shape)
y_obs = y_true + rng.normal(scale=0.01, size=y_true.shape)
psi_obs = psi_true + rng.normal(scale=0.005, size=psi_true.shape)

stacked_data = np.column_stack((t_span, x_obs, y_obs, psi_obs))

optimiser = chron.CMAES().with_max_iter(500).with_threshold(1e-10)
cost = chron.GaussianNLL(0.05)

builder = (
    chron.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(stacked_data)
    .with_tolerances(rtol=1e-6, atol=1e-8)
    .with_parameter("L", 10.0)
    .with_cost(cost)
    .with_optimiser(optimiser)
)
problem = builder.build()
results = problem.optimise()
print(results)

sampler = chron.DynamicNestedSampler().with_live_points(128)
samples = sampler.run(problem, initial=results.x)

print("time       :", samples.time)
print("log(Z)      :", samples.log_evidence)
print("information :", samples.information)
print("evaluations :", samples.draws)
