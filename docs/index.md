# Chronopt

**chron**os-**opt**imum is a Rust-first toolkit for time-series inference and optimisation with ergonomic Python bindings. It couples high-performance solvers with a highly customisable builder API for identification and optimisation of differential systems.

## Why Chronopt?

Chronopt offers a different paradigm for a parameter inference library. Conventionally, Python-based inference libraries are constructed via python bindings to a high-performance forward model with the inference algorithms implemented in Python. This package instead introduces an alternative, where the Python layer acts purely as a declarative configuration interface,
while all computationally intensive work (the optimisation / sampling loop, gradient calculations, etc.) happens entirely within the Rust runtime without crossing the FFI boundary repeatedly. This is architecture is presented visually below,

<br>

<figure markdown="span">
  ![Paradigm Comparison](chronopt.drawio.svg){ width="100%" }
  <figcaption>Conventional approach vs Chronopt: the optimisation loop moves from Python to Rust</figcaption>
</figure>


## Core Capabilities

- **Gradient-free** (Nelder-Mead, CMA-ES) and **gradient-based** (Adam) optimisers with configurable convergence criteria
- **Multi-threaded differential equation fitting** via [DiffSL](https://github.com/martinjrobins/diffsl) with dense or sparse [Diffsol](https://github.com/martinjrobins/diffsol) backends
- **Customisable likelihood/cost metrics** and Monte-Carlo sampling for posterior exploration
- **Flexible integration** with state-of-the-art differential solvers, such as [Diffrax](https://github.com/patrick-kidger/diffrax), [DifferentialEquations.jl](https://github.com/SciML/diffeqpy)

## Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __5-Minute Quickstart__

    ---

    Get started with Chronopt in 5 minutes with a simple scalar optimisation example.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-function:{ .lg .middle } __First ODE Fit__

    ---

    Learn how to fit differential equations to data using DiffSL and Diffsol.

    [:octicons-arrow-right-24: ODE Fitting Tutorial](getting-started/first-ode-fit.md)

-   :material-book-open-variant:{ .lg .middle } __Core Concepts__

    ---

    Understand the builder pattern, problem types, and optimiser vs sampler workflows.

    [:octicons-arrow-right-24: Concepts Guide](getting-started/concepts.md)

-   :material-code-tags:{ .lg .middle } __API Reference__

    ---

    Browse the complete Python and Rust API documentation.

    [:octicons-arrow-right-24: API Reference](api-reference/index.md)

</div>

## Installation

Chronopt targets Python >= 3.11. Windows builds are currently marked experimental.

=== "pip"

    ```bash
    pip install chronopt

    # Optional extras for plotting
    pip install "chronopt[plotting]"
    ```

=== "uv"

    ```bash
    uv pip install chronopt

    # Optional extras for plotting
    uv pip install "chronopt[plotting]"
    ```

## Example: Scalar Optimisation

```python
import numpy as np
import chronopt as chron

def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value])

builder = (
    chron.ScalarBuilder()
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

## Example: ODE Fitting

```python
import numpy as np
import chronopt as chron

# Logistic growth model in DiffSL
dsl = """
in_i {r = 1, k = 1 }
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
result = optimiser.run(problem, [0.5, 0.5])

print(result.x)
```

## Next Steps

<div class="grid cards" markdown>

-   [:material-school:{ .lg .middle } __Tutorials__](tutorials/index.md)

    Interactive Jupyter notebooks for hands-on learning

-   [:material-book-multiple:{ .lg .middle } __User Guides__](guides/index.md)

    In-depth guides on choosing and tuning algorithms

-   [:material-flask:{ .lg .middle } __Examples Gallery__](examples/gallery.md)

    Visual gallery of example applications

-   [:material-code-braces:{ .lg .middle } __Development__](development/index.md)

    Contributing, architecture, and building from source

</div>
