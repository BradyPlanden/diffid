"""Predator-prey parameter identification using JAX/Diffrax and Chronopt.

Demonstrates parameter estimation for the Lotka-Volterra model by:
1. Defining the predator-prey ODE in JAX
2. Generating synthetic noisy observations
3. Recovering parameters using Chronopt optimization

Prerequisites:
    pip install chronopt diffrax jax numpy
"""

from __future__ import annotations

import importlib.util
import pathlib

import chronopt as chron
import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax import config, jit

# Enable float64 precision
config.update("jax_enable_x64", True)

# Problem setup
T_SPAN = jnp.linspace(0.0, 15.0, 200)
INITIAL_STATE = jnp.array([10.0, 5.0])  # [prey, predator]
TRUE_PARAMS = np.array([1.1, 0.4, 0.1, 0.4])  # [alpha, beta, delta, gamma]


def lotka_volterra(t, state, params):
    """Lotka-Volterra predator-prey dynamics.

    dx/dt = alpha*x - beta*x*y      (prey growth and predation)
    dy/dt = delta*x*y - gamma*y     (predator growth and death)
    """
    x, y = state
    alpha, beta, delta, gamma = params
    return jnp.array([alpha * x - beta * x * y, delta * x * y - gamma * y])


# Configure ODE solver
solver = dfx.Tsit5()
saveat = dfx.SaveAt(ts=T_SPAN)
term = dfx.ODETerm(lotka_volterra)

# Extract scalar bounds for JIT compilation
t0, t1 = float(T_SPAN[0]), float(T_SPAN[-1])
dt0 = float(T_SPAN[1] - T_SPAN[0])


@jit
def simulate_jax(params):
    """JAX-native ODE integration (JIT-compiled)."""
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=INITIAL_STATE,
        args=params,
        saveat=saveat,
    )
    return sol.ys.reshape(-1)


def simulate(params):
    """NumPy wrapper for Chronopt compatibility."""
    return np.asarray(simulate_jax(jnp.asarray(params)))


# Warm up jit
_ = simulate(TRUE_PARAMS)


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
observed = data["observed_flat"]

# Parameter identification
result = (
    chron.VectorBuilder()
    .with_objective(simulate)
    .with_data(observed)
    .with_parameter("alpha", 1.3)
    .with_parameter("beta", 0.3)
    .with_parameter("delta", 0.05)
    .with_parameter("gamma", 0.6)
    .with_cost(chron.SSE())
    .with_optimiser(chron.NelderMead().with_max_iter(1000))
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
