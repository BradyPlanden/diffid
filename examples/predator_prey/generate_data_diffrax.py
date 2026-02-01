from __future__ import annotations

import os
import pathlib

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax import config, jit

config.update("jax_enable_x64", True)

T_SPAN = jnp.linspace(0.0, 15.0, 200)
INITIAL_STATE = jnp.array([10.0, 5.0])
TRUE_PARAMS = jnp.array([1.1, 0.4, 0.1, 0.4])
NOISE_STD = 0.05
SEED = 8


def lotka_volterra(_t, state, params):
    x, y = state
    alpha, beta, delta, gamma = params
    return jnp.array([alpha * x - beta * x * y, delta * x * y - gamma * y])


solver = dfx.Tsit5()
saveat = dfx.SaveAt(ts=T_SPAN)
term = dfx.ODETerm(lotka_volterra)

# scalars for JIT shape stability
t0, t1 = float(T_SPAN[0]), float(T_SPAN[-1])
dt0 = float(T_SPAN[1] - T_SPAN[0])


@jit
def simulate_jax(params):
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
    return sol.ys  # (n, 2)


def main(out_path: str | os.PathLike | None = None) -> str:
    ys = simulate_jax(TRUE_PARAMS)
    ys_np = np.asarray(ys)

    rng = np.random.default_rng(seed=SEED)
    noise = NOISE_STD * rng.standard_normal(ys_np.shape)
    ys_noisy = ys_np + noise

    t_span = np.asarray(T_SPAN)

    data = {
        "t_span": t_span,
        "initial_state": np.asarray(INITIAL_STATE),
        "true_params": np.asarray(TRUE_PARAMS),
        "y_clean": ys_np,  # (n,2)
        "observed_matrix": ys_noisy,  # (n,2)
        "observed_flat": ys_noisy.reshape(-1),  # (2n,)
        "observed_stacked": np.column_stack((t_span, ys_noisy)),  # (n,3)
        "noise_std": float(NOISE_STD),
        "seed": int(SEED),
    }

    if out_path is None:
        out_path = pathlib.Path(__file__).with_name("synthetic_data.npz")
    out_path = str(out_path)
    np.savez(out_path, **data)
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Wrote synthetic data to: {path}")
