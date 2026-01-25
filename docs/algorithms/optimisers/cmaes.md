# CMA-ES Algorithm

Covariance Matrix Adaptation Evolution Strategy (CMA-ES) is a stochastic, derivative-free algorithm for global optimisation. It maintains a multivariate Gaussian distribution that adapts its covariance structure to the objective landscape.

## Algorithm Overview

CMA-ES samples a population of candidate solutions from a multivariate normal distribution, ranks them by fitness, and updates the distribution parameters to bias future samples towards better regions.

### Key Properties

| Property | Value |
|----------|-------|
| Type | Global, gradient-free |
| Parallelisable | Yes (population evaluation) |
| Function evaluations | $\lambda$ per generation |
| Best dimensions | 10-100+ parameters |

## Algorithm

The algorithm samples offspring from:

$$
x_k \sim m + \sigma \cdot \mathcal{N}(0, C)
$$

Where:

- $m$ is the distribution mean (current best estimate)
- $\sigma$ is the global step size
- $C$ is the covariance matrix

## Algorithm Steps

1. **Sample**: Generate $\lambda$ offspring from $\mathcal{N}(m, \sigma^2 C)$
2. **Evaluate**: Compute fitness for all offspring (parallelisable)
3. **Select**: Rank offspring; select best $\mu$ as parents
4. **Update mean**: $m \leftarrow \sum_{i=1}^{\mu} w_i x_{i:\lambda}$
5. **Update evolution paths**:
    - Conjugate path: $p_\sigma \leftarrow (1-c_\sigma) p_\sigma + \sqrt{c_\sigma(2-c_\sigma)\mu_\text{eff}} \cdot C^{-1/2}(m - m_\text{old})/\sigma$
    - Covariance path: $p_c \leftarrow (1-c_c) p_c + \sqrt{c_c(2-c_c)\mu_\text{eff}} \cdot (m - m_\text{old})/\sigma$
6. **Update covariance**: Rank-one + rank-$\mu$ update
7. **Update step size**: Based on evolution path length vs expected length
8. **Repeat**: Until convergence

### Strategy Parameters

The algorithm automatically computes these from dimension $n$:

| Parameter | Formula | Purpose |
|-----------|---------|---------|
| $\lambda$ | $\max(4, \lfloor 4 + 3\ln(n) \rfloor)$ | Population size |
| $\mu$ | $\lfloor \lambda / 2 \rfloor$ | Parent count |
| $\mu_\text{eff}$ | $1 / \sum w_i^2$ | Effective parent count |
| $c_\sigma$ | $(μ_\text{eff} + 2) / (n + μ_\text{eff} + 5)$ | Step size learning rate |
| $c_c$ | $(4 + μ_\text{eff}/n) / (n + 4 + 2μ_\text{eff}/n)$ | Covariance path learning rate |


## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 1000 | Maximum generations |
| `threshold` | 1e-6 | Objective convergence tolerance |
| `step_size` | 0.5 | Initial search radius $\sigma$ |
| `population_size` | Auto | Offspring per generation $\lambda$ |
| `seed` | None | Random seed for reproducibility |
| `patience` | None | Timeout in seconds |

## Convergence Criteria

The algorithm terminates when any condition is met:

1. **Iteration limit**: `generation >= max_iter`
2. **Function tolerance**: Best objective value below `threshold`
3. **Patience**: Elapsed time exceeds `patience` seconds

## Tuning Guidance

**Step size** ($\sigma$) is the most important parameter:

- Set to ~1/3 of the expected distance to the optimum
- Too large: slow convergence, overshooting
- Too small: premature convergence, stuck in local optima

**Population size** affects exploration vs exploitation:

- Default formula works well for most problems
- Increase for highly multi-modal landscapes
- Decrease for faster convergence on simpler problems
- Match to available parallel compute resources

## When to Use

**Strengths:**

- Global search capability
- Scales well to high dimensions (10-100+ parameters)
- Self-adapting covariance learns problem structure
- Parallelisable population evaluation
- Robust to local minima

**Limitations:**

- More evaluations than gradient methods
- Stochastic results (use seed for reproducibility)
- Memory scales as $O(n^2)$ for covariance matrix
- Overkill for simple, low-dimensional problems

## Example

```python
import diffid as chron

optimiser = (
    diffid.CMAES()
    .with_max_iter(500)
    .with_step_size(0.3)
    .with_population_size(20)
    .with_seed(42)
)

result = optimiser.run(problem, initial_guess=[0.5] * n_params)
```

## Implementation Notes

- Full covariance matrix (not diagonal approximation)
- Lazy eigendecomposition for efficient sampling
- Bounds enforced via clamping after each sample

## References

1. Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial". arXiv:1604.00772.
2. Hansen, N. and Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies". *Evolutionary Computation*, 9(2), 159-195.
3. Hansen, N. et al. (2003). "Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)". *Evolutionary Computation*, 11(1), 1-18.

## See Also

- [API Reference](../../api-reference/python/optimisers.md#cma-es)
- [Choosing an Optimiser](../../guides/choosing-optimiser.md)
- [Parallel Execution](../../guides/parallel-execution.md)
