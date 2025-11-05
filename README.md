# chronopt

**chron**os-**opt**imum is a Rust-first toolkit for time-series inference and optimisation with ergonomic Python bindings. It couples high-performance solvers with builder APIs for modelling objective functions and differential systems.

## Project goals
- Speed and numerical accuracy through a Rust core.
- Modular components with informative diagnostics.
- Batteries-included experience spanning optimisation, sampling, and plotting.

## Key capabilities
- Deterministic optimisers (Nelder-Mead, CMA-ES) with configurable stopping criteria.
- Differential equation fitting via [DiffSL](https://github.com/martinjrobins/diffsl) with dense or sparse [Diffsol](https://github.com/martinjrobins/diffsol) backends.
- Customisable likelihood/cost metrics and Monte-Carlo sampling for posterior exploration.
- Flexible integration with common differential solvers, such as [Diffrax](https://github.com/patrick-kidger/diffrax), [DifferentialEquations.jl](https://github.com/SciML/diffeqpy)
- Python builder APIs mirroring the Rust core plus generated type stubs for autocompletion.

## Installation

Chronopt targets Python 3.11 or newer.

```bash
pip install chronopt

# Or with uv
uv pip install chronopt

# Optional extras
pip install "chronopt[plotting]"
```

## Quickstart (Python)

```python
import numpy as np
import chronopt as chron


def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value], dtype=float)


builder = (
    chron.PythonBuilder()
    .with_callable(rosenbrock)
    .with_parameter("x", 1.5)
    .with_parameter("y", -1.5)
)
problem = builder.build()
result = problem.optimize()

print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.fun:.3e}")
print(f"Success: {result.success}")
```

### Differential solver workflow

```python
import numpy as np
import chronopt as chron


# Example diffsol ODE (logistic growth)
dsl = """
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

t = np.linspace(0.0, 5.0, 51)
observations = np.exp(-1.3 * t)
data = np.column_stack((t, observations))

builder = (
    chron.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_parameter("k", 1.0)
    .with_backend("dense")
)
problem = builder.build()

optimiser = chron.CMAES().with_max_iter(1000)
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
uv run cargo run -p chronopt-py --no-default-features --features stubgen --bin generate_stubs
```

Without `uv`, invoke the generator directly:

```bash
cargo run -p chronopt-py --no-default-features --features stubgen --bin generate_stubs
```

### Pre-commit hooks

```bash
uv tool install pre-commit
pre-commit install
pre-commit run --all-files
```

### Tests

```bash
uv sync && uv run pytest  # Python tests
cargo test                 # Rust tests
```