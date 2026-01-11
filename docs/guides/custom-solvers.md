# Custom Solvers Guide

!!! info "Coming Soon"
    This guide is being written. Check back soon for Diffrax and DifferentialEquations.jl integration examples.

## Overview

Use `VectorBuilder` to integrate custom ODE solvers:
- JAX/Diffrax
- Julia/DifferentialEquations.jl
- Custom Python solvers

## Basic Pattern

```python
def custom_solver(params):
    # Your solver here
    # Return predictions at observation times
    return predictions

builder = (
    chron.VectorBuilder()
    .with_objective(custom_solver)
    .with_data(data)
    .with_parameter("alpha", 1.0)
)
```

## Examples

See the predator-prey examples:
- [predator_prey_diffrax.py](https://github.com/bradyplanden/chronopt/blob/main/examples/predator_prey/predator_prey_diffrax.py)
- [predator_prey_diffeqpy.py](https://github.com/bradyplanden/chronopt/blob/main/examples/predator_prey/predator_prey_diffeqpy.py)

## See Also

- [VectorBuilder API](../api-reference/python/builders.md#vectorbuilder)
- [Examples Gallery](../examples/gallery.md)
