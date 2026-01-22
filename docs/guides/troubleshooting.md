# Troubleshooting Guide

### Import Error

```python
ImportError: No module named 'chronopt'
```

**Solution**: Install Chronopt: `pip install chronopt`

### Poor Fit Quality

**Solutions:**

1. Try different optimisers
2. Increase iterations, i.e. `.with_max_iter(10000)`
3. Try different initial conditions
4. Normalise data if scales vary widely

### Slow Performance

**Solutions:**

1. Use CMA-ES for parallelisation
2. Reduce the number of data points
3. Use sparse backend for large ODE systems

## See Also

- [Installation](../getting-started/installation.md)
- [Tuning Optimisers](tuning-optimizers.md)
- [GitHub Issues](https://github.com/bradyplanden/chronopt/issues)
