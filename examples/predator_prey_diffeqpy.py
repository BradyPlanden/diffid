"""Predator-prey parameter identification using DifferentialEquations.jl and Chronopt.

Demonstrates parameter estimation for the Lotka-Volterra model by:
1. Defining the predator-prey ODE in Julia via diffeqpy
2. Generating synthetic noisy observations
3. Recovering parameters using Chronopt optimization

Prerequisites:
    pip install diffeqpy chronopt numpy
    python -c "from diffeqpy import de; de.install()"
"""

from __future__ import annotations

import chronopt as chron
import numpy as np
from diffeqpy import de

# Problem setup
T_SPAN = np.linspace(0.0, 15.0, 200)
INITIAL_STATE = np.array([10.0, 5.0])  # [prey, predator]
TRUE_PARAMS = np.array([1.1, 0.4, 0.1, 0.4])  # [alpha, beta, delta, gamma]


def lotka_volterra(state, params, _t):
    """Lotka-Volterra predator-prey dynamics.

    dx/dt = alpha*x - beta*x*y      (prey growth and predation)
    dy/dt = delta*x*y - gamma*y     (predator growth and death)
    """
    x, y = state
    alpha, beta, delta, gamma = params
    return [alpha * x - beta * x * y, delta * x * y - gamma * y]


# Configure Julia ODE problem
ode_problem = de.ODEProblem(
    lotka_volterra,
    INITIAL_STATE.tolist(),
    (float(T_SPAN[0]), float(T_SPAN[-1])),
    tuple(TRUE_PARAMS),
)
solver = de.Tsit5()


def simulate(params):
    """Simulate the ODE system with given parameters."""
    prob = de.remake(ode_problem, p=tuple(params))
    sol = de.solve(prob, solver, saveat=T_SPAN)
    return np.vstack([np.array(state) for state in sol.u]).ravel()


# Generate synthetic observations with noise
rng = np.random.default_rng(seed=8)
observed = simulate(TRUE_PARAMS) + 0.05 * rng.standard_normal(len(T_SPAN) * 2)

# Parameter identification
result = (
    chron.VectorBuilder()
    .with_objective(simulate)
    .with_data(observed)
    .with_parameter("alpha", 0.8)
    .with_parameter("beta", 0.3)
    .with_parameter("delta", 0.05)
    .with_parameter("gamma", 0.6)
    .with_cost(chron.cost.SSE())
    .with_optimiser(chron.NelderMead().with_max_iter(1000))
    .build()
    .optimize()
)

# Display results
print("Parameter Identification Results")
print("-" * 50)
print(f"True parameters:      {TRUE_PARAMS}")
print(f"Estimated parameters: {np.array(result.x)}")
print(f"Final SSE:            {result.fun:.6f}")
print(
    f"Relative error:       {np.linalg.norm(result.x - TRUE_PARAMS) / np.linalg.norm(TRUE_PARAMS):.2%}"
)
