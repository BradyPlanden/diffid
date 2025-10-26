# chronopt

Chronopt is a time-series statistical inference package, it's goals are:
- Be fast, without sacrificing safety
- Be modular and informative

## Installation 
```bash
pip install chronopt
```

## Quickstart (Pure Python)
```python
import numpy as np
import chronopt as chron


def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value], dtype=float)


# Build the optimisation problem
builder = (
    chron.PythonBuilder()
    .add_callable(rosenbrock)
    .add_parameter("x")
    .add_parameter("y")
)
problem = builder.build()

# Choose an optimiser and solve
optimiser = chron.NelderMead().with_max_iter(500).with_threshold(1e-6)
result = optimiser.run(problem, [1.5, -1.5])

print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.fun:.3e}")
print(f"Success: {result.success}")
```

## Development Installation
Clone this repository, python installation via:
```bash
uv sync
```

Building the rust package w/ python bindings:
```bash
uv run maturin develop
```

Regenerate the Python *.pyi stub files after making changes to the bindings:
```bash
uv run cargo run -p chronopt-py --no-default-features --features stubgen --bin generate_stubs
```

## Tests
To run the python tests, use pytest:
```bash
uv run pytest
```

for the rust tests, use cargo:
```bash
cargo test
```