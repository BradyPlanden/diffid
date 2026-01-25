# Results

Result objects returned by optimisers and samplers containing optimal parameters, diagnostics, and metadata.

## OptimisationResults

::: diffid.OptimisationResults
    options:
      show_root_heading: true
      show_source: false

All optimisers return an `OptimisationResults` object with the following attributes:

### Example Usage

```python
import diffid as chron

problem = builder.build()
result = problem.optimise()

# Access results
print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")
print(f"Success: {result.success}")
print(f"Iterations: {result.iterations}")
print(f"Evaluations: {result.evaluations}")
print(f"Message: {result.message}")
```

---

## Attributes

### `x`

**Type:** `numpy.ndarray`

Optimal parameter values found by the optimiser.

```python
result.x  # e.g., array([1.0, 2.5, 0.8])
```

Parameters are ordered according to `.with_parameter()` calls:

```python
builder = (
    diffid.ScalarBuilder()
    .with_parameter("alpha", 1.0)  # result.x[0]
    .with_parameter("beta", 2.0)   # result.x[1]
)
```

---

### `value`

**Type:** `float`

Objective function value at the optimum (`result.x`).

```python
result.value  # e.g., 1.234e-08
```

For minimisation problems, lower values are better.

---

### `success`

**Type:** `bool`

Indicates whether optimisation terminated successfully.

```python
if result.success:
    print("Optimisation converged")
else:
    print(f"Failed: {result.message}")
```

**`True`** when:

- Convergence criteria met
- Threshold reached
- Normal termination

**`False`** when:

- Maximum iterations exceeded without convergence
- Numerical errors occurred
- User-requested termination

---

### `iterations`

**Type:** `int`

Number of iterations performed.

```python
result.iterations  # e.g., 157
```

- **Nelder-Mead**: Number of simplex iterations
- **CMA-ES**: Number of generations
- **Adam**: Number of gradient steps

---

### `evaluations`

**Type:** `int`

Total number of objective function evaluations.

```python
result.evaluations  # e.g., 314
```

Note that `evaluations` can be much larger than `iterations`:

- **CMA-ES**: `evaluations ≈ iterations × population_size`
- **Nelder-Mead**: `evaluations ≈ iterations × (n_params + 1)`
- **Adam**: `evaluations ≈ iterations` (one per step)

---

### `message`

**Type:** `str`

Human-readable termination message.

```python
result.message
# e.g., "Converged successfully"
# e.g., "Maximum iterations reached"
# e.g., "Threshold achieved"
```

Use this for debugging and logging.

---

## Common Patterns

### Checking Success

```python
result = problem.optimise()

if result.success:
    print(f"Found optimum: {result.x}")
    print(f"Objective: {result.value:.3e}")
else:
    print(f"Optimisation failed: {result.message}")
    print(f"Best found: {result.x} (value: {result.value:.3e})")
```

Even when `success=False`, `result.x` contains the best parameters found.

---

### Extracting Parameters by Name

Results don't store parameter names. Track them manually if needed:

```python
param_names = ["alpha", "beta", "gamma"]

builder = diffid.ScalarBuilder().with_objective(func)
for name in param_names:
    builder = builder.with_parameter(name, 1.0)

problem = builder.build()
result = problem.optimise()

# Create a dictionary
params = dict(zip(param_names, result.x))
print(params)  # {'alpha': 1.05, 'beta': 2.31, 'gamma': 0.87}
```

---

### Comparing Multiple Runs

```python
initial_guesses = [[1, 1], [-1, -1], [0, 2]]

results = [
    optimiser.run(problem, guess)
    for guess in initial_guesses
]

# Find best result
best = min(results, key=lambda r: r.value)
print(f"Best from {len(results)} runs: {best.x}, value={best.value:.3e}")
```

---

### Assessing Convergence

```python
result = optimiser.run(problem, initial_guess)

# Check various indicators
converged = (
    result.success
    and result.value < 1e-6
    and result.iterations < max_iters
)

if not converged:
    print(f"Warning: May not have converged")
    print(f"  Iterations: {result.iterations}")
    print(f"  Value: {result.value:.3e}")
    print(f"  Message: {result.message}")
```

---

## Sampler Results (Future)

When samplers are fully implemented, they will return different result types:

### MCMC Results

```python
result = mcmc_sampler.run(problem, initial_guess)

result.samples        # (n_samples, n_params) array
result.log_likelihood  # Log-likelihood values
result.acceptance_rate  # For diagnostics
```

### Nested Sampling Results

```python
result = nested_sampler.run(problem, initial_guess)

result.log_evidence      # Log marginal likelihood
result.evidence_error    # Uncertainty in evidence
result.samples          # Posterior samples
result.weights          # Sample weights
```

---

## Diagnostics

### Visualising Convergence

```python
import matplotlib.pyplot as plt

# Run optimisation with tracking (custom wrapper)
values = []

def tracked_func(x):
    val = original_func(x)
    values.append(val[0])
    return val

# ... run optimisation ...

plt.plot(values)
plt.xlabel('Evaluation')
plt.ylabel('Objective Value')
plt.yscale('log')
plt.title('Optimisation Convergence')
plt.show()
```

### Checking Parameter Bounds

```python
result = problem.optimise()

# Check if parameters are in reasonable range
for i, val in enumerate(result.x):
    if abs(val) > 1e3:
        print(f"Warning: Parameter {i} has large magnitude: {val}")
```

---

## Performance Metrics

```python
import time

start = time.time()
result = optimiser.run(problem, initial_guess)
elapsed = time.time() - start

print(f"Total time: {elapsed:.2f}s")
print(f"Evaluations: {result.evaluations}")
print(f"Time per evaluation: {elapsed/result.evaluations*1000:.2f}ms")
```

---

## See Also

- [Optimisers](optimisers.md)
- [Samplers](samplers.md)
- [Choosing an Optimiser](../../guides/choosing-optimiser.md)
- [Troubleshooting](../../guides/troubleshooting.md)
