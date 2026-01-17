# Core Concepts

This guide explains the fundamental concepts and patterns in Chronopt.

## The Builder Pattern

Chronopt uses the **builder pattern** for constructing problems. This provides a fluent, chainable API for configuration:

```python
builder = (
    chron.ScalarBuilder()
    .with_objective(my_function)
    .with_parameter("x", 1.0)
    .with_parameter("y", 2.0)
)
problem = builder.build()
```

**Benefits:**

- **Clear and readable**: Method names clearly describe what's being configured
- **Flexible**: Add components in any order
- **Type-safe**: Catch errors early with proper type hints
- **Chainable**: Fluent interface for concise code

## Problem Types

Chronopt provides different builders for different problem types:

### ScalarBuilder

For **direct function optimisation** where you have a Python callable:

```python
def objective(x):
    return np.asarray([x[0]**2 + x[1]**2])

problem = (
    chron.ScalarBuilder()
    .with_objective(objective)
    .with_parameter("x", 0.0)
    .with_parameter("y", 0.0)
    .build()
)
```

**Use when:**
- You have a direct Python function to minimise
- No differential equations involved
- Simple parameter optimisation

### DiffsolBuilder

For **ODE parameter fitting** using the built-in DiffSL/Diffsol solver:

```python
dsl = """
in_i {r = 1, k = 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""

problem = (
    chron.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_parameter("k", 1.0)
    .with_backend("dense")
    .build()
)
```

**Use when:**
- Fitting ODE parameters to time-series data
- Using DiffSL for model definition
- Need high-performance multi-threaded solving

### VectorBuilder

For **custom ODE solvers** (Diffrax, DifferentialEquations.jl, etc.):

```python
def solve_ode(params):
    # Your custom ODE solver
    # Returns predictions at observation times
    return predictions

problem = (
    chron.VectorBuilder()
    .with_objective(solve_ode)
    .with_data(data)
    .with_parameter("alpha", 1.0)
    .with_parameter("beta", 0.5)
    .build()
)
```

**Use when:**
- Need a specific solver (JAX/Diffrax, Julia/DifferentialEquations.jl)
- Complex ODEs not supported by DiffSL
- Custom forward models beyond ODEs

See the [Custom Solvers Guide](../guides/custom-solvers.md) for examples.

## Parameters

Parameters are the decision variables you want to optimise:

```python
builder = (
    chron.ScalarBuilder()
    .with_parameter("x", initial_value=1.0)  # Name and initial guess
    .with_parameter("y", initial_value=-1.0)
)
```

**Important:**

- Parameters must have unique names
- Initial values are required
- Order matters: results will be returned in the same order

## Optimisers vs Samplers

Chronopt provides two types of algorithms:

### Optimisers: Finding the Best Solution

**Goal**: Find parameter values that minimise the objective function.

**Algorithms**:

- **Nelder-Mead**: Gradient-free, local search
- **CMA-ES**: Gradient-free, global search
- **Adam**: Gradient-based (automatic differentiation)

**Usage:**

```python
# Default optimiser (Nelder-Mead)
result = problem.optimise()

# Specific optimiser
optimiser = chron.CMAES().with_max_iter(1000)
result = optimiser.run(problem, initial_guess)
```

**Returns:** A single best solution

### Samplers: Exploring Uncertainty

**Goal**: Sample from the posterior distribution to quantify parameter uncertainty.

**Algorithms**:

- **Metropolis-Hastings**: MCMC sampling for posterior exploration
- **Dynamic Nested Sampling**: Evidence calculation for model comparison

**Usage:**

```python
sampler = chron.MetropolisHastings().with_max_iter(10000)
result = sampler.run(problem, initial_guess)

# Result contains samples, not a single optimum
print(result.samples.shape)  # (n_samples, n_parameters)
```

**Returns:** A collection of samples from the posterior

**When to use:**

| Use Optimisers When | Use Samplers When |
|---------------------|-------------------|
| You want the single best fit | You want to quantify uncertainty |
| Point estimates are sufficient | You need confidence intervals |
| Computation budget is limited | You need full posterior distributions |
| | Comparing multiple models (Bayes factors) |

See [Choosing an Optimiser](../guides/choosing-optimiser.md) and [Choosing a Sampler](../guides/choosing-sampler.md) for detailed guidance.

## The Ask/Tell Pattern

For advanced use cases, Chronopt supports the **ask/tell pattern** for manual control of the optimisation loop:

```python
optimiser = chron.CMAES().with_max_iter(1000)

# Ask for candidates
candidates = optimiser.ask(n_candidates=10)

# Evaluate them (potentially in parallel or on remote machines)
evaluations = [problem.evaluate(c) for c in candidates]

# Tell the optimiser the results
optimiser.tell(candidates, evaluations)

# Repeat until convergence
while not optimiser.should_stop():
    candidates = optimiser.ask()
    evaluations = [problem.evaluate(c) for c in candidates]
    optimiser.tell(candidates, evaluations)

result = optimiser.get_result()
```

**Use cases:**

- Distributed optimisation across multiple machines
- Custom evaluation pipelines
- Hybrid optimisation strategies
- Integration with external simulators

## Cost Metrics

Cost metrics define how model predictions are compared to observations:

```python
from chronopt import SSE, RMSE, GaussianNLL

# Sum of squared errors (default)
builder = builder.with_cost_metric(SSE())

# Root mean squared error (normalised)
builder = builder.with_cost_metric(RMSE())

# Gaussian negative log-likelihood (for probabilistic inference)
builder = builder.with_cost_metric(GaussianNLL())
```

**Common metrics:**

- **SSE** (Sum of Squared Errors): Standard least squares, sensitive to outliers
- **RMSE** (Root Mean Squared Error): Normalised by number of points
- **GaussianNLL**: For Bayesian inference and sampling

See the [Cost Metrics Guide](../guides/cost-metrics.md) for more details.

## Results

All optimisers and samplers return result objects with standard attributes:

### Optimiser Results

```python
result = problem.optimise()

print(result.x)           # Optimal parameters (NumPy array)
print(result.value)       # Objective value at optimum (float)
print(result.success)     # Whether optimisation succeeded (bool)
print(result.iterations)  # Number of iterations (int)
print(result.evaluations) # Number of function evaluations (int)
print(result.message)     # Termination message (str)
```

### Sampler Results

```python
result = sampler.run(problem, initial_guess)

print(result.samples)     # MCMC samples (NumPy array, shape: (n_samples, n_params))
print(result.log_likelihood)  # Log-likelihood values
print(result.acceptance_rate) # Acceptance rate (for diagnostics)
```

For nested sampling:

```python
result = dns_sampler.run(problem, initial_guess)

print(result.log_evidence)     # Log marginal likelihood
print(result.evidence_error)   # Uncertainty in evidence
print(result.samples)          # Posterior samples
```

## Parallelisation

Chronopt automatically parallelises where possible:

- **DiffsolBuilder**: Multi-threaded ODE solving
- **CMA-ES**: Parallel candidate evaluation
- **Dynamic Nested Sampling**: Parallel live point evaluation

Control parallelism:

```python
# Limit threads for ODE solving
builder = builder.with_max_threads(4)

# Population size for CMA-ES (larger = more parallel work)
optimiser = chron.CMAES().with_population_size(20)
```

See the [Parallel Execution Guide](../guides/parallel-execution.md) for details.

## Key Takeaways

- **Builders** provide a fluent API for problem construction
- **ScalarBuilder** for direct functions, **DiffsolBuilder** for ODEs, **VectorBuilder** for custom solvers
- **Parameters** are decision variables with names and initial values
- **Optimisers** find the best solution; **samplers** explore uncertainty
- **Cost metrics** define how predictions are compared to observations
- **Ask/tell pattern** enables advanced control and distributed computation

## Next Steps

- **[Tutorials](../tutorials/index.md)**: Interactive Jupyter notebooks
- **[Choosing an Optimiser](../guides/choosing-optimiser.md)**: Learn when to use each algorithm
- **[API Reference](../api-reference/index.md)**: Browse complete API documentation
- **[Examples Gallery](../examples/gallery.md)**: Visual gallery of applications
