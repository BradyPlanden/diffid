# Builders

Builders provide a fluent API for constructing optimisation problems. Choose the appropriate builder based on your problem type.

## Overview

| Builder | Use Case | Example |
|---------|----------|---------|
| `ScalarBuilder` | Direct function optimisation | Rosenbrock, Rastrigin |
| `DiffsolBuilder` | ODE fitting with DiffSL | Parameter identification |
| `VectorBuilder` | Custom solver integration | JAX, Julia, external simulators |

## ScalarBuilder

::: chronopt.ScalarBuilder
    options:
      show_root_heading: true
      show_source: false
      members:
        - __new__
        - with_objective
        - with_parameter
        - with_cost_metric
        - build

### Example Usage

```python
import numpy as np
import chronopt as chron

def rosenbrock(x):
    return np.asarray([(1 - x[0])**2 + 100*(x[1] - x[0]**2)**2])

builder = (
    chron.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", 1.5)
    .with_parameter("y", -1.5)
)
problem = builder.build()
result = problem.optimise()
```

### When to Use

- You have a Python function to minimise directly
- No differential equations involved
- Simple parameter optimisation or test functions

---

## DiffsolBuilder

::: chronopt.DiffsolBuilder
    options:
      show_root_heading: true
      show_source: false
      members:
        - __new__
        - with_diffsl
        - with_data
        - with_parameter
        - with_backend
        - with_cost_metric
        - with_max_threads
        - build

### Example Usage

```python
import numpy as np
import chronopt as chron

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
result = optimiser.run(problem, [0.5, 0.5])
```

### Backend Options

- **`"dense"`** (default): For systems with < 100 variables, dense Jacobian
- **`"sparse"`**: For large systems (> 100 variables), sparse Jacobian

### DiffSL Syntax

DiffSL is a domain-specific language for ODEs:

```
in = [param1, param2]           # Parameters to fit
param1 { default1 }             # Default values
param2 { default2 }
u_i { state1 = init1 }          # Initial conditions
F_i { derivative_expr }         # dy/dt expressions
out_i { state1, state2 }        # Optional: output variables
```

### When to Use

- Fitting ODE parameters to time-series data
- Using Chronopt's built-in high-performance solver
- Models expressible in DiffSL syntax

---

## VectorBuilder

::: chronopt.VectorBuilder
    options:
      show_root_heading: true
      show_source: false
      members:
        - __new__
        - with_objective
        - with_data
        - with_parameter
        - with_cost_metric
        - build

### Example Usage

```python
import numpy as np
import chronopt as chron

def custom_solver(params):
    """Your custom ODE solver (e.g., using JAX/Diffrax)."""
    # Solve ODE with params
    # Return predictions at observation times
    return predictions  # NumPy array

# Observed data
t = np.linspace(0, 10, 100)
observations = ...  # Your experimental data
data = np.column_stack((t, observations))

builder = (
    chron.VectorBuilder()
    .with_objective(custom_solver)
    .with_data(data)
    .with_parameter("alpha", 1.0)
    .with_parameter("beta", 0.5)
)
problem = builder.build()
result = problem.optimise()
```

### Callable Requirements

Your callable must:

1. Accept a NumPy array of parameters
2. Return a NumPy array of predictions
3. Predictions must match observation times in the data

### When to Use

- Need specific solver features (stiff solvers, event detection, etc.)
- Using JAX/Diffrax for automatic differentiation
- Using Julia/DifferentialEquations.jl via diffeqpy
- Custom forward models beyond ODEs (PDEs, agent-based models, etc.)

See the [Custom Solvers Guide](../../guides/custom-solvers.md) for examples with Diffrax and DifferentialEquations.jl.

---

## Problem

::: chronopt.Problem
    options:
      show_root_heading: true
      show_source: false

The `Problem` class is created by calling `.build()` on a builder. It represents a fully configured optimisation problem.

### Common Methods

```python
# Optimise with default settings (Nelder-Mead)
result = problem.optimise()

# Evaluate objective at specific parameters
value = problem.evaluate(params)
```

---

## Common Patterns

### Chaining Methods

Builders use a fluent interface - chain methods in any order:

```python
builder = (
    chron.ScalarBuilder()
    .with_objective(func)
    .with_parameter("x", 1.0)
    .with_parameter("y", 2.0)
    .with_cost_metric(chron.RMSE())
)
```

### Reusing Builders

Builders are immutable - each method returns a new builder:

```python
base = chron.ScalarBuilder().with_objective(func)

problem1 = base.with_parameter("x", 1.0).build()
problem2 = base.with_parameter("x", 2.0).build()  # Different initial guess
```

### Parameter Order Matters

Parameters are indexed in the order they're added:

```python
builder = (
    chron.ScalarBuilder()
    .with_parameter("y", 2.0)  # Index 0
    .with_parameter("x", 1.0)  # Index 1
)

result = problem.optimise()
print(result.x)  # [optimal_y, optimal_x]
```

## See Also

- [Getting Started: Core Concepts](../../getting-started/concepts.md)
- [Cost Metrics](cost-metrics.md)
- [Optimisers](optimizers.md)
- [Custom Solvers Guide](../../guides/custom-solvers.md)
