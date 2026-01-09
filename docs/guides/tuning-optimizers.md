# Tuning Optimisers

!!! info "Coming Soon"
    This guide is being written. Check back soon for parameter tuning strategies.

## Quick Tips

### Nelder-Mead

- **step_size**: Start with 10-50% of parameter range
- **threshold**: Use 1e-6 for standard precision
- **max_iter**: Use `100 * n_parameters` as minimum

### CMA-ES

- **step_size**: Start with ~1/3 of expected parameter range
- **population_size**: Default formula works well, increase for more exploration
- **max_iter**: Each iteration evaluates `population_size` candidates

### Adam

- **step_size**: Most critical - try 0.1, 0.01, 0.001, 0.0001
- **betas**: Defaults (0.9, 0.999) usually work well

## See Also

- [Optimisers API](../api-reference/python/optimizers.md)
- [Choosing an Optimiser](choosing-optimizer.md)
- [Troubleshooting](troubleshooting.md)
