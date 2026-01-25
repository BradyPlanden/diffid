# Cost Metrics Guide

!!! info "Coming Soon"
    This guide is being written. Check back soon for detailed cost metric guidance.

## Quick Reference

| Metric | Use Case |
|--------|----------|
| SSE | Standard least squares |
| RMSE | Normalised error, model comparison |
| GaussianNLL | Bayesian inference, sampling |

## Choosing a Metric

- **SSE**: Default for most optimisation
- **RMSE**: When comparing models with different data sizes
- **GaussianNLL**: Required for samplers (MCMC, nested sampling)

## See Also

- [Cost Metrics API](../api-reference/python/cost-metrics.md)
- [Samplers](../api-reference/python/samplers.md)
