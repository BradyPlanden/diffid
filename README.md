<div align="center">

# diffid
[![Python Versions from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fbradyplanden%2Fdiffid%2Fmain%2Fpyproject.toml&label=Python)](https://pypi.org/project/diffid/)
[![License](https://img.shields.io/github/license/bradyplanden/diffid?color=blue)](https://github.com/bradyplanden/diffid/blob/main/LICENSE)
[![Releases](https://img.shields.io/github/v/release/bradyplanden/diffid?color=gold)](https://github.com/bradyplanden/diffid/releases)

**diff**erential **id**entification offers a different paradigm for a parameter inference library. Conventional Python-based inference libraries are constructed via python bindings to a high-performance forward model with the optimisation algorithms implemented in Python.  Diffid offers an alternative, where the Python layer acts purely as a declarative configuration interface, while all computationally intensive work (the optimisation / sampling loop, gradient calculations, etc.) happens entirely within the Rust runtime without crossing the FFI boundary repeatedly. This is architecture is presented visually below,

</div>

<p align="center">
    <img src="https://raw.githubusercontent.com/BradyPlanden/diffid/main/docs/assets/diffid.svg" alt="Diffid architecture describing the two optimisation approaches." width="700" />
</p>

## Project goals
- Speed and numerical accuracy through a Rust core.
- Modular components with informative diagnostics.
- Batteries-included experience spanning optimisation, sampling, and plotting.

## Core capabilities
- Gradient-free (Nelder-Mead, CMA-ES) and gradient-based (Adam) optimisers with configurable convergence criteria.
- Multi-threaded differential equation fitting via [DiffSL](https://github.com/martinjrobins/diffsl) with dense or sparse [Diffsol](https://github.com/martinjrobins/diffsol) backends.
- Customisable likelihood/cost metrics and Monte-Carlo sampling for posterior exploration.
- Flexible integration with state-of-the-art differential solvers, such as [Diffrax](https://github.com/patrick-kidger/diffrax), [DifferentialEquations.jl](https://github.com/SciML/diffeqpy)

## Installation

Diffid targets Python >= 3.11. Windows builds are currently marked experimental.

```bash
pip install diffid

# Or with uv
uv pip install diffid

# Optional extras
pip install "diffid[plotting]"
```

## Quickstart

### ScalarProblem
```python
import numpy as np
import diffid


def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value])


builder = (
    diffid.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", 1.5)
    .with_parameter("y", -1.5)
)
problem = builder.build()
result = problem.optimise()

print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")
print(f"Success: {result.success}")
```

### Differential solver workflow

```python
import numpy as np
import diffid


# Example diffsol ODE (logistic growth)
dsl = """
in_i { r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

t = np.linspace(0.0, 5.0, 51)
observations = np.exp(-1.3 * t)
data = np.column_stack((t, observations))

builder = (
    diffid.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_parameter("k", 1.0)
    .with_backend("dense")
)
problem = builder.build()

optimiser = diffid.CMAES().with_max_iter(1000)
result = optimiser.run(problem, [0.5,0.5])

print(result.x)
```

## Development setup

Clone the repository and create the Python environment:

```bash
uv sync
```

Build the Rust extension with Python bindings:

```bash
uv run maturin develop
```

Regenerate `.pyi` stubs after changing the bindings:

```bash
uv run cargo run -p diffid-py --no-default-features --features stubgen --bin generate_stubs
```

Without `uv`, invoke the generator directly:

```bash
cargo run -p diffid-py --no-default-features --features stubgen --bin generate_stubs
```

### Pre-commit hooks

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

### Tests

```bash
uv run maturin develop && uv run pytest -v # Python tests
cargo test # Rust tests
```