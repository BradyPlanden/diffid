# Metropolis-Hastings Algorithm

Metropolis-Hastings is a Markov Chain Monte Carlo (MCMC) algorithm for sampling from posterior distributions. It enables uncertainty quantification by generating samples that characterise the probability distribution over parameters.

## Algorithm Overview

The algorithm constructs a Markov chain whose stationary distribution is the target posterior. By proposing random moves and accepting/rejecting them based on the likelihood ratio, it explores the parameter space in proportion to posterior probability.

### Key Properties

| Property | Value |
|----------|-------|
| Type | MCMC sampler |
| Parallelisable | Yes (across chains) |
| Output | Posterior samples |
| Best for | Uncertainty quantification |

## Mathematical Foundation

Given a target distribution $\pi(\theta)$ (the posterior), the algorithm:

1. Proposes a new state from proposal distribution $q(\theta' | \theta)$
2. Computes acceptance probability:

$$
\alpha = \min\left(1, \frac{\pi(\theta') q(\theta | \theta')}{\pi(\theta) q(\theta' | \theta)}\right)
$$

For the symmetric random walk proposal used here ($q(\theta' | \theta) = q(\theta | \theta')$):

$$
\alpha = \min\left(1, \frac{\pi(\theta')}{\pi(\theta)}\right) = \min\left(1, \exp(\log\pi(\theta') - \log\pi(\theta))\right)
$$

## Algorithm Steps

1. **Initialise**: Start at initial point $\theta_0$
2. **Propose**: Generate candidate $\theta' = \theta_t + \sigma \cdot \mathcal{N}(0, I)$
3. **Evaluate**: Compute log-likelihood $\log\mathcal{L}(\theta')$
4. **Accept/Reject**:
    - Draw $u \sim \text{Uniform}(0, 1)$
    - If $u < \exp(\log\mathcal{L}(\theta') - \log\mathcal{L}(\theta_t))$: accept $\theta_{t+1} = \theta'$
    - Else: reject $\theta_{t+1} = \theta_t$
5. **Repeat**: For specified number of iterations
6. **Return**: Chain of samples $\{\theta_0, \theta_1, \ldots, \theta_T\}$

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | 1000 | MCMC iterations per chain |
| `num_chains` | 1 | Number of independent chains |
| `step_size` | 0.1 | Proposal distribution width $\sigma$ |
| `seed` | None | Random seed for reproducibility |

## Tuning Guidance

**Step size** controls exploration efficiency:

- Target acceptance rate: 20-50% (optimal ~23% for high dimensions)
- Too large: low acceptance rate, chain gets stuck
- Too small: high acceptance rate but slow mixing
- Tune by monitoring acceptance ratio

**Number of chains** aids convergence diagnostics:

- Multiple chains from different starting points detect convergence issues
- Enables Gelman-Rubin $\hat{R}$ diagnostic
- More chains = more robust inference but higher cost

**Iterations** should be sufficient for:

- Burn-in period (discard initial samples)
- Effective sample size for reliable estimates

## Convergence Diagnostics

After sampling, assess convergence:

- **Trace plots**: Visual inspection for stationarity
- **Acceptance rate**: Should be 20-50%
- **Autocorrelation**: Lower is better; high autocorrelation means inefficient sampling
- **$\hat{R}$ statistic**: Should be < 1.1 for all parameters (requires multiple chains)

## When to Use

**Strengths:**

- Full posterior characterisation
- Uncertainty quantification (credible intervals)
- Parameter correlation analysis
- Simple and robust

**Limitations:**

- Many function evaluations required
- Can be slow for high-dimensional problems
- Requires careful tuning for efficiency
- Does not compute model evidence (use nested sampling)

## Cost Metric Requirement

Metropolis-Hastings requires a **negative log-likelihood** cost metric:

```python
problem = (
    chron.ScalarProblemBuilder()
    .with_objective(model_fn)
    .with_cost_metric(chron.CostMetric.GaussianNLL)  # Required
    .build()
)
```

## Example

```python
import chronopt as chron

sampler = (
    chron.MetropolisHastings()
    .with_iterations(5000)
    .with_num_chains(4)
    .with_step_size(0.05)
    .with_seed(42)
)

result = sampler.run(problem, initial_guess=[1.0, 2.0])

# Access samples
samples = result.samples  # Shape: (n_chains * iterations, n_params)
```

## References

1. Metropolis, N. et al. (1953). "Equation of State Calculations by Fast Computing Machines". *Journal of Chemical Physics*, 21(6), 1087-1092.
2. Hastings, W.K. (1970). "Monte Carlo Sampling Methods Using Markov Chains and Their Applications". *Biometrika*, 57(1), 97-109.
3. Gelman, A. et al. (2013). *Bayesian Data Analysis*. 3rd ed. Chapman & Hall/CRC.

## See Also

- [API Reference](../../api-reference/python/samplers.md)
- [Choosing a Sampler](../../guides/choosing-sampler.md)
- [Cost Metrics](../../guides/cost-metrics.md)
