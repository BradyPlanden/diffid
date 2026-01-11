# User Guides

In-depth guides for making the most of Chronopt's optimisation and sampling capabilities.

## Algorithm Selection

<div class="grid cards" markdown>

-   :material-tune:{ .lg .middle } __Choosing an Optimiser__

    ---

    Decision trees and guidelines for selecting the right optimisation algorithm.

    [:octicons-arrow-right-24: Choosing an Optimiser](choosing-optimiser.md)

-   :material-speedometer:{ .lg .middle } __Tuning Optimisers__

    ---

    Parameter tuning strategies and troubleshooting for each algorithm.

    [:octicons-arrow-right-24: Tuning Guide](tuning-optimizers.md)

-   :material-chart-bell-curve:{ .lg .middle } __Choosing a Sampler__

    ---

    When to use MCMC vs nested sampling for uncertainty quantification.

    [:octicons-arrow-right-24: Choosing a Sampler](choosing-sampler.md)

</div>

## Problem Configuration

<div class="grid cards" markdown>

-   :material-function-variant:{ .lg .middle } __Cost Metrics__

    ---

    Understanding SSE, RMSE, and GaussianNLL for different use cases.

    [:octicons-arrow-right-24: Cost Metrics Guide](cost-metrics.md)

-   :material-script:{ .lg .middle } __DiffSL Backend__

    ---

    Choosing between dense and sparse solvers for ODE systems.

    [:octicons-arrow-right-24: DiffSL Backend Guide](diffsol-backend.md)

-   :material-connection:{ .lg .middle } __Custom Solvers__

    ---

    Integrating Diffrax, DifferentialEquations.jl, and other external solvers.

    [:octicons-arrow-right-24: Custom Solvers](custom-solvers.md)

</div>

## Performance

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } __Parallel Execution__

    ---

    Thread safety, parallelisation strategies, and performance optimisation.

    [:octicons-arrow-right-24: Parallel Execution](parallel-execution.md)

</div>

## Troubleshooting

<div class="grid cards" markdown>

-   :material-help-circle:{ .lg .middle } __Troubleshooting__

    ---

    Common errors, solutions, and debugging strategies.

    [:octicons-arrow-right-24: Troubleshooting Guide](troubleshooting.md)

</div>

## Quick Reference

### When to Use Each Component

| Component | Use Case |
|-----------|----------|
| **ScalarBuilder** | Direct function optimisation |
| **DiffsolBuilder** | ODE fitting with DiffSL |
| **VectorBuilder** | Custom solvers (JAX, Julia) |
| **Nelder-Mead** | Local search, < 10 parameters |
| **CMA-ES** | Global search, 10-100+ parameters |
| **Adam** | Gradient-based, smooth objectives |
| **SSE** | Standard least squares |
| **RMSE** | Normalised error |
| **GaussianNLL** | Bayesian inference |

### Typical Workflows

**Simple Optimisation:**
```
ScalarBuilder → optimise() → Result
```

**ODE Fitting:**
```
DiffsolBuilder → CMAES → Result → Visualise
```

**Uncertainty Quantification:**
```
DiffsolBuilder + GaussianNLL → Optimise → MCMC → Posterior
```

**Model Comparison:**
```
Multiple models + GaussianNLL → Nested Sampling → Evidence → Bayes factors
```

## See Also

- [Getting Started](../getting-started/index.md) - Core concepts and installation
- [Tutorials](../tutorials/index.md) - Interactive notebooks
- [API Reference](../api-reference/index.md) - Complete API documentation
- [Algorithms](../algorithms/index.md) - Algorithm-specific documentation
