<div align="center" markdown>

# Diffid

<b>diff</b>erential <b>id</b>entification is a Rust-first toolkit for time-series inference and optimisation with ergonomic Python bindings. It couples high-performance solvers with a highly customisable builder API for identification and optimisation of differential systems.

</div>

## Why Diffid?

Diffid offers a different paradigm for a parameter inference library. Conventionally, Python-based inference libraries are constructed via python bindings to a high-performance forward model with the inference algorithms implemented in Python. Alongside this approach, Diffid introduces an alternative, where the Python layer acts purely as a declarative configuration interface, while all computationally intensive work (the optimisation / sampling loop, gradient calculations, etc.) happens entirely within the Rust runtime without crossing the FFI boundary repeatedly. This is architecture is presented visually below,

<br>

<figure markdown="span">
  ![Paradigm Comparison](assets/diffid.svg){ width="100%" }
  <figcaption>Conventional vs configuration approach: the optimisation loop moves from Python to Rust</figcaption>
</figure>


## Core Capabilities

<div class="grid cards" markdown>

-   __Optimisation Algorithms__

    ---

    Gradient-free (Nelder-Mead, CMA-ES) and gradient-based (Adam) optimisers with configurable convergence criteria

-   __High-Performance ODE Fitting__

    ---

    Multi-threaded differential equation fitting via [DiffSL](https://github.com/martinjrobins/diffsl) with dense or sparse [Diffsol](https://github.com/martinjrobins/diffsol) backends

-   __Uncertainty Quantification__

    ---

    Customisable likelihood/cost metrics and Monte-Carlo sampling for posterior exploration

-   __Flexible Integration__

    ---

    Integration with state-of-the-art differential solvers: [Diffrax](https://github.com/patrick-kidger/diffrax), [DifferentialEquations.jl](https://github.com/SciML/diffeqpy)

</div>

## Installation

Diffid targets Python >= 3.11. Windows builds are currently marked experimental.

=== "pip"

    ```bash
    pip install diffid

    # Optional extras for plotting
    pip install "diffid[plotting]"
    ```

=== "uv"

    ```bash
    uv pip install diffid

    # Optional extras for plotting
    uv pip install "diffid[plotting]"
    ```

## Example: Scalar Optimisation

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

## Example: ODE Fitting

```python
import numpy as np
import diffid

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
    diffid.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_parameter("k", 1.0)
    .with_backend("dense")
)
problem = builder.build()

optimiser = diffid.CMAES().with_max_iter(1000)
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
