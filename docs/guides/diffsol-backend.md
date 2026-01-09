# DiffSL Backend Guide

!!! info "Coming Soon"
    This guide is being written. Check back soon for detailed backend selection guidance.

## Quick Reference

| Backend | Best For |
|---------|----------|
| `"dense"` | < 100 state variables |
| `"sparse"` | > 100 state variables, sparse Jacobian |

## Usage

```python
builder = (
    chron.DiffsolBuilder()
    .with_diffsl(dsl)
    .with_data(data)
    .with_backend("dense")  # or "sparse"
)
```

## See Also

- [DiffsolBuilder API](../api-reference/python/builders.md#diffsolbuilder)
- [First ODE Fit](../getting-started/first-ode-fit.md)
