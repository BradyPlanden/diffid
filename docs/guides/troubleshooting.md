# Troubleshooting Guide

### Import Error

```python
ImportError: No module named 'diffid'
```

**Solution**: Install Diffid: `pip install diffid`

### Poor Fit Quality

**Solutions:**

1. Try different optimisers
2. Increase iterations, i.e. `.with_max_iter(10000)`
3. Try different initial conditions
4. Normalise data if scales vary widely

### Optimiser Terminates With `FunctionEvaluationFailed`

This indicates your objective (or solver backend) returned an evaluation error.

**Solutions:**

1. Validate your objective is finite for the explored parameter region
2. Add/adjust bounds to exclude unstable regions
3. Check solver configuration tolerances and initial conditions (Diffsol problems)
4. Re-run with logging around your model evaluation path to locate the failing input

### Bounds Dimension Mismatch

If `initial_guess` dimension and bounds dimension differ, optimiser initialisation fails fast.

**Solutions:**

1. Ensure every optimised parameter has a matching bound tuple
2. Ensure `initial_guess` length matches builder parameter count
3. Verify custom Ask/Tell loops reuse consistent dimensionality

### Slow Performance

**Solutions:**

1. Use CMA-ES for parallelisation
2. Reduce the number of data points
3. Use sparse backend for large ODE systems

## See Also

- [Installation](../getting-started/installation.md)
- [Tuning Optimisers](tuning-optimisers.md)
- [GitHub Issues](https://github.com/bradyplanden/diffid/issues)
