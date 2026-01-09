# Adam Algorithm

!!! info "Coming Soon"
    Detailed algorithm documentation is being written.

## Overview

Adaptive Moment Estimation (Adam) is a gradient-based optimiser with adaptive learning rates.

## When to Use

**Best for:**
- Smooth, differentiable objectives
- Fast convergence needed
- Gradients available or cheap to compute

**Avoid when:**
- Objective is non-smooth
- No gradient information
- Need global optimum (can get stuck in local minima)

## API Reference

See [Adam API](../../api-reference/python/optimizers.md#adam) for usage details.

## See Also

- [Choosing an Optimiser](../../guides/choosing-optimizer.md)
- [Tuning Optimisers](../../guides/tuning-optimizers.md)
