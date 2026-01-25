# Parallel Execution Guide

!!! info "Coming Soon"
    This guide is being written. Check back soon for parallelisation strategies.

## Quick Tips

- **DiffsolBuilder**: Automatically multi-threaded
- **CMA-ES**: Parallel population evaluation
- **Dynamic Nested Sampling**: Parallel live point evaluation

## Controlling Threads

```python
# Limit threads for ODE solving
builder = builder.with_max_threads(4)

# Population size for CMA-ES
optimiser = diffid.CMAES().with_population_size(20)
```

## See Also

- [DiffsolBuilder API](../api-reference/python/builders.md#diffsolbuilder)
- [CMA-ES API](../api-reference/python/optimisers.md#cma-es)
