# API Reference

Complete API documentation for Diffid's Python and Rust interfaces.

## Python API

Diffid provides a comprehensive Python API with full type hints and automatic documentation.

<div class="grid cards" markdown>

-   :material-hammer-wrench:{ .lg .middle } __Builders__

    ---

    Problem builders for different use cases: scalar functions, ODE fitting, and custom solvers.

    [:octicons-arrow-right-24: Builders](python/builders.md)

-   :material-tune:{ .lg .middle } __Optimisers__

    ---

    Gradient-free (Nelder-Mead, CMA-ES) and gradient-based (Adam) optimisation algorithms.

    [:octicons-arrow-right-24: Optimisers](python/optimisers.md)

-   :material-chart-bell-curve:{ .lg .middle } __Samplers__

    ---

    MCMC and nested sampling for posterior exploration and model comparison.

    [:octicons-arrow-right-24: Samplers](python/samplers.md)

-   :material-function-variant:{ .lg .middle } __Cost Metrics__

    ---

    Metrics for comparing model predictions to observations (SSE, RMSE, GaussianNLL).

    [:octicons-arrow-right-24: Cost Metrics](python/cost-metrics.md)

-   :material-chart-line:{ .lg .middle } __Results__

    ---

    Result objects returned by optimisers and samplers with diagnostics.

    [:octicons-arrow-right-24: Results](python/results.md)

</div>

## Rust API

The Rust core provides high-performance implementations of all algorithms.

[:octicons-arrow-right-24: Rust Documentation on docs.rs](https://docs.rs/diffid/latest/diffid/)

## Module Structure

```
diffid/
├── ScalarBuilder        # Direct function optimisation
├── DiffsolBuilder       # ODE fitting with DiffSL/Diffsol
├── VectorBuilder        # Custom solver integration
├── Problem              # Built problem instance
├── Optimisers
│   ├── NelderMead       # Simplex gradient-free optimiser
│   ├── CMAES            # Covariance matrix adaptation
│   └── Adam             # Adaptive moment estimation
├── Samplers
│   ├── MetropolisHastings  # MCMC sampling
│   └── DynamicNestedSampling  # Evidence calculation
├── CostMetric           # Cost/likelihood metrics
└── OptimisationResults  # Result container
```

## Type Hints

All Python functions include comprehensive type hints:

```python
from diffid import ScalarBuilder, CMAES, OptimisationResults
import numpy.typing as npt

def optimize_function(
    func: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    initial_params: dict[str, float]
) -> OptimisationResults:
    builder = ScalarBuilder().with_objective(func)
    for name, value in initial_params.items():
        builder = builder.with_parameter(name, value)

    problem = builder.build()
    optimiser = CMAES().with_max_iter(1000)
    return optimiser.run(problem, list(initial_params.values()))
```

Type stubs are automatically generated from the Rust implementation and are available in IDE completions.

## Conventions

### Parameter Order

All builders maintain parameter order based on the sequence of `.with_parameter()` calls:

```python
builder = (
    diffid.ScalarBuilder()
    .with_parameter("x", 1.0)  # Index 0
    .with_parameter("y", 2.0)  # Index 1
)

result = problem.optimise()
print(result.x)  # [optimal_x, optimal_y]
```

### Return Types

- **Optimisers** return `OptimisationResults` with `.x`, `.value`, `.success`, etc.
- **Samplers** return sampling-specific results with `.samples`, `.log_likelihood`, etc.
- All results include diagnostic information

### Array Conventions

- **Input**: Python callables accept NumPy arrays
- **Output**: Python callables must return NumPy arrays
- **Data**: Data arrays are `(n_timepoints, n_variables + 1)` with time in first column

## Next Steps

- [Builders API](python/builders.md) - Start building problems
- [Optimisers API](python/optimisers.md) - Configure optimisation algorithms
- [User Guides](../guides/index.md) - Learn when to use each component
- [Tutorials](../tutorials/index.md) - Interactive examples
