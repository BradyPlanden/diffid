# Nelder-Mead Algorithm

!!! info "Coming Soon"
    Detailed algorithm documentation is being written.

## Overview

The Nelder-Mead algorithm (also known as the downhill simplex method) is a gradient-free local optimisation algorithm.

## When to Use

**Best for:**

- Problems with < 10 parameters
- Noisy objective functions
- No gradient information available
- Quick exploration

**Avoid when:**

- More than 10 parameters
- Need global optimum
- Very tight convergence required

## API Reference

See [NelderMead API](../../api-reference/python/optimizers.md#nelder-mead) for usage details.

## See Also

- [Choosing an Optimiser](../../guides/choosing-optimiser.md)
- [Tuning Optimisers](../../guides/tuning-optimizers.md)
