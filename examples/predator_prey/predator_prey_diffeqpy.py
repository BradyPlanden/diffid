"""Predator-prey parameter identification using DifferentialEquations.jl and diffid.

Demonstrates parameter estimation for the Lotka-Volterra model by:
1. Defining the predator-prey ODE in Julia via diffeqpy
2. Generating synthetic noisy observations
3. Recovering parameters using diffid optimisation

Prerequisites:
    pip install diffeqpy diffid numpy
    python -c "from diffeqpy import de; de.install()"
"""

from __future__ import annotations

import importlib.util
import pathlib

import diffid
import numpy as np
from diffeqpy import de

# Problem setup
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
    (0.0, 15.0),
    tuple(TRUE_PARAMS),
)
solver = de.Tsit5()


def simulate(params):
    """Simulate the ODE system with given parameters."""
    prob = de.remake(ode_problem, p=tuple(params))
    sol = de.solve(prob, solver, saveat=T_SPAN)
    return np.vstack([np.array(state) for state in sol.u]).ravel()


# Load shared synthetic observations (generate if missing)
data_path = pathlib.Path(__file__).with_name("synthetic_data.npz")
if not data_path.exists():
    gen_path = pathlib.Path(__file__).with_name("generate_data_diffrax.py")
    spec = importlib.util.spec_from_file_location("pp_gen", gen_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module.main(data_path)
data = np.load(str(data_path))
T_SPAN = data["t_span"]
observed = data["observed_flat"]

# Parameter identification
result = (
    diffid.VectorBuilder()
    .with_objective(simulate)
    .with_data(observed)
    .with_parameter("alpha", 0.8)
    .with_parameter("beta", 0.3)
    .with_parameter("delta", 0.05)
    .with_parameter("gamma", 0.6)
    .with_cost(diffid.SSE())
    .with_optimiser(diffid.NelderMead().with_max_iter(1000))
    .build()
    .optimise()
)

# Display results
print("Parameter Identification Results")
print("-" * 50)
print(f"True parameters:      {TRUE_PARAMS}")
print(f"Estimated parameters: {np.array(result.x)}")
print(f"Final SSE:            {result.value:.6f}")
print(
    f"Relative error:       {np.linalg.norm(result.x - TRUE_PARAMS) / np.linalg.norm(TRUE_PARAMS):.2%}"
)
print(f"Success:              {result.success}")
print(f"Iterations:           {result.iterations}")
print(f"Time:                 {result.time}s")
