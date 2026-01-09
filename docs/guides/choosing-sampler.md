# Choosing a Sampler

!!! info "Coming Soon"
    This guide is being written. Sampler Python bindings are also in development.

## Quick Reference

| Use Case | Sampler |
|----------|---------|
| Uncertainty quantification | Metropolis-Hastings |
| Model comparison | Dynamic Nested Sampling |
| Just posterior samples | Metropolis-Hastings |
| Need evidence/Bayes factors | Dynamic Nested Sampling |

## When to Sample vs Optimize

Use **optimisers** when:
- Point estimate is sufficient
- Speed is critical
- Don't need uncertainty

Use **samplers** when:
- Need confidence intervals
- Comparing models
- Want full posterior

## See Also

- [Samplers API](../api-reference/python/samplers.md)
- [Optimisers vs Samplers](../getting-started/concepts.md#optimisers-vs-samplers)
