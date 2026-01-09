# Troubleshooting Guide

!!! info "Coming Soon"
    This guide is being written. Check back soon for common issues and solutions.

## Quick Fixes

### Import Error

```python
ImportError: No module named 'chronopt'
```

**Solution**: Install Chronopt: `pip install chronopt`

### Poor Fit Quality

**Solutions:**
1. Try different optimisers (CMA-ES is often more robust)
2. Increase iterations: `.with_max_iter(10000)`
3. Check initial conditions
4. Normalise data if scales vary widely

### Slow Performance

**Solutions:**
1. Use CMA-ES for parallelisation
2. Reduce data points if possible
3. Use sparse backend for large ODE systems
4. Profile with `py-spy` to find bottlenecks

## See Also

- [Installation](../getting-started/installation.md)
- [Tuning Optimisers](tuning-optimizers.md)
- [GitHub Issues](https://github.com/bradyplanden/chronopt/issues)
