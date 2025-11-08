"""Minimal predator-prey parameter identification using JAX/Diffrax and Chronopt.

This example demonstrates how to:
1. Define an ODE system (Lotka-Volterra predator-prey model)
2. Generate synthetic noisy observations
3. Set up a parameter identification problem using Chronopt
4. Estimate unknown parameters from the observations
"""

from __future__ import annotations

import chronopt as chron
import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray
from jax import config as jax_config
from jax import device_get

# Enable float64 precision for numerical accuracy
jax_config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Problem configuration
# ---------------------------------------------------------------------------

# Time span and initial conditions
T_SPAN = jnp.linspace(0.0, 15.0, 200, dtype=jnp.float64)
INITIAL_STATE = jnp.array([10.0, 5.0], dtype=jnp.float64)  # [prey, predator]

# True parameters: [alpha, beta, delta, gamma]
# alpha: prey growth rate, beta: predation rate
# delta: predator efficiency, gamma: predator mortality rate
TRUE_PARAMS = np.array([1.1, 0.4, 0.1, 0.4], dtype=np.float64)

# Integration parameters
T0 = float(T_SPAN[0])
T1 = float(T_SPAN[-1])
DT0 = float(T_SPAN[1] - T_SPAN[0])

# ---------------------------------------------------------------------------
# ODE system definition
# ---------------------------------------------------------------------------


def lotka_volterra(t: float, y: JaxArray, params: JaxArray) -> JaxArray:
    """Lotka-Volterra predator-prey dynamics.

    Args:
        t: Current time (unused in autonomous system)
        y: State vector [prey, predator]
        params: Parameter vector [alpha, beta, delta, gamma]

    Returns:
        Derivative vector [dprey/dt, dpredator/dt]
    """
    alpha, beta, delta, gamma = params
    prey, predator = y
    return jnp.array(
        [
            alpha * prey - beta * prey * predator,  # Prey dynamics
            delta * prey * predator - gamma * predator,  # Predator dynamics
        ],
        dtype=jnp.float64,
    )


# Configure the ODE solver
ODE_TERM = dfx.ODETerm(lotka_volterra)
SAVE_AT = dfx.SaveAt(ts=T_SPAN)
SOLVER = dfx.Tsit5()  # Fifth-order Runge-Kutta method

# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def simulate(x: np.ndarray) -> np.ndarray:
    """Integrate the Lotka-Volterra system for given parameters.

    Args:
        x: Parameter vector [alpha, beta, delta, gamma]

    Returns:
        Flattened state trajectory [prey_0, pred_0, prey_1, pred_1, ...]
    """
    params = jnp.asarray(x, dtype=jnp.float64)
    solution = dfx.diffeqsolve(
        ODE_TERM,
        SOLVER,
        t0=T0,
        t1=T1,
        dt0=DT0,
        y0=INITIAL_STATE,
        args=params,
        saveat=SAVE_AT,
    )
    return np.asarray(device_get(solution.ys.reshape(-1)), dtype=np.float64)


# ---------------------------------------------------------------------------
# Generate synthetic observations
# ---------------------------------------------------------------------------

# Create clean observations from true parameters
_observed_clean = simulate(TRUE_PARAMS)

# Add measurement noise (5% Gaussian)
_rng = np.random.default_rng(seed=8)
_noise = 0.05 * _rng.standard_normal(_observed_clean.shape)
OBSERVED_VECTOR = _observed_clean + _noise

# ---------------------------------------------------------------------------
# Parameter identification setup and solve
# ---------------------------------------------------------------------------

# Configure optimizer
optimiser = chron.NelderMead().with_max_iter(1_000)

# Build the identification problem
builder = (
    chron.VectorBuilder()
    .with_objective(simulate)  # Forward model
    .with_data(OBSERVED_VECTOR)  # Target observations
    .with_parameter("alpha", 0.8)  # Initial guess for prey growth
    .with_parameter("beta", 0.3)  # Initial guess for predation rate
    .with_parameter("delta", 0.05)  # Initial guess for predator efficiency
    .with_parameter("gamma", 0.6)  # Initial guess for predator mortality
    .with_cost(chron.cost.SSE())  # Sum of squared errors cost function
    .with_optimiser(optimiser)
)

problem = builder.build()
result = problem.optimize()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

print("Parameter Identification Results")
print("-" * 40)
print(f"True parameters:      {TRUE_PARAMS}")
print(f"Estimated parameters: {np.asarray(result.x)}")
print(f"Final SSE:            {result.fun:.6f}")
