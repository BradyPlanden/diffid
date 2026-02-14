# Nelder-Mead Algorithm

The Nelder-Mead algorithm (downhill simplex method) is a gradient-free local optimisation algorithm that uses a geometric simplex to navigate the parameter space.

## Algorithm Overview

Nelder-Mead maintains a **simplex** of $n+1$ vertices in $n$-dimensional space. At each iteration, it transforms the simplex through reflection, expansion, contraction, or shrinking operations to move towards lower objective values.

### Key Properties

| Property | Value |
|----------|-------|
| Type | Local, gradient-free |
| Parallelisable | No |
| Function evaluations | Sequential |
| Best dimensions | 2-10 parameters |

## Algorithm Steps

1. **Initialise**: Create simplex from initial point using step size
2. **Order**: Sort vertices by objective value: $f(x_1) \leq f(x_2) \leq \cdots \leq f(x_{n+1})$
3. **Reflect**: Compute reflection point $x_r = \bar{x} + \alpha(\bar{x} - x_{n+1})$
4. **Transform**: Based on $f(x_r)$:
    - If $f(x_1) \leq f(x_r) < f(x_n)$: accept reflection
    - If $f(x_r) < f(x_1)$: try expansion $x_e = \bar{x} + \gamma(x_r - \bar{x})$
    - If $f(x_r) \geq f(x_n)$: try contraction $x_c = \bar{x} + \rho(x_{n+1} - \bar{x})$
5. **Shrink**: If contraction fails, shrink towards best vertex
6. **Repeat**: Until convergence criteria met

Where $\bar{x}$ is the centroid of all vertices except the worst.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 1000 | Maximum iterations |
| `threshold` | 1e-6 | Objective convergence tolerance |
| `step_size` | 0.1 | Initial simplex size |
| `position_tolerance` | 1e-6 | Parameter space convergence tolerance |
| `patience` | None | Timeout in seconds |

### Simplex Coefficients

| Coefficient | Symbol | Default | Purpose |
|-------------|--------|---------|---------|
| Reflection | $\alpha$ | 1.0 | Scale of reflection step |
| Expansion | $\gamma$ | 2.0 | Scale of expansion step |
| Contraction | $\rho$ | 0.5 | Scale of contraction step |
| Shrinking | $\sigma$ | 0.5 | Scale of shrink operation |

## Convergence Criteria

The algorithm terminates when any of the following conditions is met:

1. **Iteration limit**: `iteration >= max_iter`
2. **Function tolerance**: Change in objective value below `threshold`
3. **Position tolerance**: Simplex diameter below `position_tolerance`
4. **Patience**: Elapsed time exceeds `patience` seconds

## Tuning Guidance

**Step size** controls the initial simplex scale:

- Set to ~10-50% of the expected parameter range
- Larger values explore more broadly
- Smaller values for local refinement near a known solution

**Thresholds** control precision vs speed:

- Tighter tolerances (1e-8) for high precision
- Looser tolerances (1e-4) for quick estimates

## When to Use

**Strengths:**

- No gradient computation required
- Robust to moderate noise
- Simple and reliable for small problems
- Low memory footprint

**Limitations:**

- Convergence slows significantly beyond 10 parameters
- Can converge to local minima
- Not parallelisable

## Failure and Validation Behaviour

- Evaluation failures terminate immediately with `FunctionEvaluationFailed`
- Bounds dimension mismatch is rejected during initialisation (fails fast)
- Internally uses a single-point ask path while preserving batch-shaped Ask/Tell compatibility

## Example

```python
import diffid as chron

optimiser = (
    diffid.NelderMead()
    .with_max_iter(5000)
    .with_step_size(0.1)
    .with_threshold(1e-8)
)

result = optimiser.run(problem, initial_guess=[1.0, 2.0])
```

## References

1. Nelder, J.A. and Mead, R. (1965). "A Simplex Method for Function Minimization". *The Computer Journal*, 7(4), 308-313.
2. Lagarias, J.C. et al. (1998). "Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions". *SIAM Journal on Optimization*, 9(1), 112-147.

## See Also

- [API Reference](../../api-reference/python/optimisers.md#nelder-mead)
- [Choosing an Optimiser](../../guides/choosing-optimiser.md)
- [Tuning Optimisers](../../guides/tuning-optimisers.md)
