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
- **history**: Leave off (`with_history(False)`) for lower overhead, enable only for diagnostics

## Failure Policy

All optimisers now use strict evaluation-failure semantics:

- Objective/solver evaluation errors terminate with `FunctionEvaluationFailed`
- Invalid bounds dimensionality fails fast during optimiser initialisation
- Invalid optimiser configuration fails fast with `FunctionEvaluationFailed`
  instead of silently clamping or ignoring invalid values

## See Also

- [Optimisers API](../api-reference/python/optimisers.md)
- [Choosing an Optimiser](choosing-optimiser.md)
- [Troubleshooting](troubleshooting.md)
