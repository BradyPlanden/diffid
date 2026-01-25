# Dynamic Nested Sampling Algorithm

Dynamic Nested Sampling is a Bayesian inference algorithm that computes the model evidence (marginal likelihood) while simultaneously generating posterior samples. It is particularly suited for model comparison via Bayes factors. 

## Algorithm Overview

Nested sampling transforms the multi-dimensional evidence integral into a one-dimensional integral over prior volume. It maintains a set of "live points" that progressively shrink the prior volume while tracking the likelihood threshold. The "dynamic" aspect allows the algorithm to allocate more live points in regions that contribute most to either the evidence or posterior, improving efficiency.

### Key Properties

| Property | Value |
|----------|-------|
| Type | Evidence sampler |
| Parallelisable | Yes (batch proposals) |
| Output | Evidence + posterior samples |
| Best for | Model comparison |

## Mathematical Foundation

The model evidence (marginal likelihood) is:

$$
\mathcal{Z} = \int \mathcal{L}(\theta) \pi(\theta) \, d\theta
$$

Nested sampling transforms this by defining the prior volume:

$$
X(\lambda) = \int_{\mathcal{L}(\theta) > \lambda} \pi(\theta) \, d\theta
$$

The evidence becomes a one-dimensional integral:

$$
\mathcal{Z} = \int_0^1 \mathcal{L}(X) \, dX
$$

This is approximated by iteratively shrinking the prior volume:

$$
\mathcal{Z} \approx \sum_{i=1}^{N} \mathcal{L}_i \, \Delta X_i
$$

Where $\Delta X_i = X_{i-1} - X_i$ and the shrinkage ratio is estimated statistically.

## Algorithm Steps

1. **Initialise**: Sample $K$ live points uniformly from the prior
2. **Iterate**:
    - Find lowest-likelihood live point $\mathcal{L}^*$
    - Record point as "dead point" with prior volume estimate
    - Replace with new point sampled uniformly from prior with $\mathcal{L} > \mathcal{L}^*$
3. **Terminate**: When remaining evidence contribution is negligible
4. **Compute**: Sum contributions to estimate $\mathcal{Z}$ and uncertainty
5. **Return**: Log-evidence, evidence error, and posterior samples

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `live_points` | 64 | Number of live points $K$ |
| `expansion_factor` | 0.5 | Live set expansion aggressiveness |
| `termination_tol` | 1e-3 | Evidence convergence tolerance |
| `mcmc_batch_size` | 8 | Proposals generated per iteration |
| `mcmc_step_size` | 0.01 | MCMC proposal step size |
| `seed` | None | Random seed for reproducibility |

## Tuning Guidance

**Live points** control accuracy vs cost:

- More live points = better evidence estimate but more evaluations
- Minimum recommended: $25 × n_{\text{params}}$
- For accurate posteriors: $>50 × n_{\text{params}}$

**Termination tolerance** affects precision:

- Smaller values = more accurate evidence but more iterations
- Default 1e-3 sufficient for most model comparisons

**MCMC parameters** affect replacement efficiency:

- `mcmc_batch_size`: larger batches for parallel evaluation
- `mcmc_step_size`: tune for ~20-50% acceptance in constrained sampling

## Evidence Interpretation

The log-evidence can be used for model comparison via Bayes factors:

$$
\ln B_{12} = \ln \mathcal{Z}_1 - \ln \mathcal{Z}_2
$$

| $\ln B_{12}$ | $B_{12}$ | Evidence strength |
|--------------|----------|-------------------|
| < 1 | < 3 | Inconclusive |
| 1-2.5 | 3-12 | Positive |
| 2.5-5 | 12-150 | Strong |
| > 5 | > 150 | Decisive |

## When to Use

**Strengths:**

- Computes model evidence directly
- Handles multi-modal posteriors well
- Provides posterior samples as byproduct
- Robust to complex likelihood landscapes
- Parallelisable

**Limitations:**

- More expensive than MCMC for posterior-only inference
- Constrained sampling can be challenging in high dimensions
- Evidence accuracy depends on live point count

## Cost Metric Requirement

Nested sampling requires a **negative log-likelihood** cost metric:

```python
problem = (
    diffid.ScalarProblemBuilder()
    .with_objective(model_fn)
    .with_cost_metric(diffid.CostMetric.GaussianNLL)  # Required
    .build()
)
```

The objective function should return negative log-likelihood; the algorithm internally negates to work with log-likelihood.

## Example

```python
import diffid as chron

sampler = (
    diffid.DynamicNestedSampling()
    .with_live_points(128)
    .with_termination_tol(1e-3)
    .with_seed(42)
)

result = sampler.run(problem, initial_guess=[1.0, 2.0])

# Access results
log_evidence = result.log_evidence
evidence_error = result.evidence_error
samples = result.samples
```

## Implementation Notes

- Uses MCMC for constrained prior sampling
- Batch evaluation for parallel efficiency
- Automatic termination based on remaining evidence contribution

## References

1. Skilling, J. (2006). "Nested Sampling for General Bayesian Computation". *Bayesian Analysis*, 1(4), 833-860.
2. Higson, E. et al. (2019). "Dynamic Nested Sampling: An Improved Algorithm for Parameter Estimation and Evidence Calculation". *Statistics and Computing*, 29, 891-913.
3. Speagle, J.S. (2020). "dynesty: A Dynamic Nested Sampling Package for Estimating Bayesian Posteriors and Evidences". *Monthly Notices of the Royal Astronomical Society*, 493(3), 3132-3158.

## See Also

- [API Reference](../../api-reference/python/samplers.md)
- [Choosing a Sampler](../../guides/choosing-sampler.md)
- [Cost Metrics](../../guides/cost-metrics.md)
