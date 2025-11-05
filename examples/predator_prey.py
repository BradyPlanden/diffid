"""Minimal predator-prey identification example using JAX and Chronopt."""

from __future__ import annotations

import chronopt as chron
import jax.numpy as jnp
import numpy as np
from jax import config as jax_config
from jax import jit
from jax.experimental.ode import odeint

# Enable float64 to match Chronopt's default precision.
jax_config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

T_SPAN = jnp.linspace(0.0, 15.0, 200)
INITIAL_STATE = jnp.array([10.0, 5.0])
TRUE_PARAMS = jnp.array([1.1, 0.4, 0.1, 0.4])


def lotka_volterra(y: jnp.ndarray, _: float, params: jnp.ndarray) -> jnp.ndarray:
    """Lotka-Volterra predator-prey dynamics."""
    alpha, beta, delta, gamma = params
    prey, predator = y
    return jnp.array(
        [
            alpha * prey - beta * prey * predator,
            delta * prey * predator - gamma * predator,
        ],
        dtype=jnp.float64,
    )


@jit
def simulate(params: jnp.ndarray) -> jnp.ndarray:
    """Integrate the Lotka-Volterra system for the supplied parameters."""
    return odeint(lotka_volterra, INITIAL_STATE, T_SPAN, params)


# Generate synthetic observations with mild Gaussian noise.
_observed_clean = np.asarray(simulate(TRUE_PARAMS))
_rng = np.random.default_rng(seed=8)
_observed_noisy = _observed_clean + 0.05 * _rng.standard_normal(_observed_clean.shape)
OBSERVED_JNP = jnp.asarray(_observed_noisy, dtype=jnp.float64)


@jit
def sse_cost(params: jnp.ndarray) -> jnp.float64:
    """Sum of squared errors between simulation and noisy observations."""
    simulated = simulate(params)
    residuals = simulated - OBSERVED_JNP
    return jnp.sum(residuals * residuals)


def objective(x: np.ndarray) -> float:
    """Wrap the SSE cost for Chronopt's PythonBuilder."""
    params = jnp.asarray(x, dtype=jnp.float64)
    return float(sse_cost(params))


def build_problem() -> chron.Problem:
    """Create the Chronopt problem for predator-prey identification."""
    builder = (
        chron.PythonBuilder()
        .with_callable(objective)
        .with_parameter("alpha")
        .with_parameter("beta")
        .with_parameter("delta")
        .with_parameter("gamma")
        .with_optimiser(chron.NelderMead().with_max_iter(1000))
    )
    return builder.build()


def main() -> None:
    problem = build_problem()
    initial_guess = [0.8, 0.3, 0.05, 0.6]
    result = problem.optimize(initial=initial_guess)

    print("True parameters:", np.asarray(TRUE_PARAMS))
    print("Estimated parameters:", np.asarray(result.x))
    print("Final SSE:", result.fun)


if __name__ == "__main__":
    main()
