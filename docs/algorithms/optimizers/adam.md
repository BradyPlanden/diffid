# Adam Algorithm

Adaptive Moment Estimation (Adam) is a first-order gradient-based optimiser that maintains adaptive learning rates for each parameter using estimates of first and second moments of the gradients.

## Algorithm Overview

Adam combines the benefits of AdaGrad (adapting to sparse gradients) and RMSprop (adapting to non-stationary objectives) by tracking exponential moving averages of both the gradient and squared gradient.

### Key Properties

| Property | Value |
|----------|-------|
| Type | Local, gradient-based |
| Parallelisable | No |
| Function evaluations | 1 per iteration |
| Best for | Smooth, differentiable objectives |

## Mathematical Foundation

At each iteration $t$, Adam updates parameters $\theta$ using:

**Gradient computation:**

$$
g_t = \nabla_\theta f(\theta_{t-1})
$$

**Biased moment estimates:**
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$$

**Bias correction:**
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
$$
$$
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

**Parameter update:**
$$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

Where $\alpha$ is the learning rate (step size).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 1000 | Maximum iterations |
| `step_size` | 0.01 | Learning rate $\alpha$ |
| `beta1` | 0.9 | First moment decay rate $\beta_1$ |
| `beta2` | 0.999 | Second moment decay rate $\beta_2$ |
| `eps` | 1e-8 | Numerical stability constant $\epsilon$ |
| `threshold` | 1e-6 | Objective convergence tolerance |
| `gradient_threshold` | None | Gradient norm convergence tolerance |
| `patience` | None | Timeout in seconds |

## Convergence Criteria

The algorithm terminates when any condition is met:

1. **Iteration limit**: `iteration >= max_iter`
2. **Gradient norm**: $\|g_t\| <$ `gradient_threshold`
3. **Objective change**: Change in objective below `threshold`
4. **Patience**: Elapsed time exceeds `patience` seconds

## Tuning Guidance

**Learning rate** (`step_size`) is the most critical parameter:

- Start with 0.001 (the original paper's recommendation)
- Try orders of magnitude: 0.1, 0.01, 0.001, 0.0001
- Too large: oscillation, divergence, or overshooting
- Too small: slow convergence

**Beta parameters** rarely need adjustment:

- $\beta_1 = 0.9$: controls momentum (gradient smoothing)
- $\beta_2 = 0.999$: controls adaptive scaling (variance smoothing)
- Lower $\beta_1$ for less momentum, more responsiveness
- Lower $\beta_2$ for faster adaptation to gradient scale changes

**Epsilon** almost never needs tuning:

- Prevents division by zero when gradients are near zero
- Default 1e-8 works for most cases

## When to Use

**Strengths:**

- Fast convergence on smooth objectives
- Per-parameter adaptive learning rates
- Handles sparse gradients well
- Works with noisy gradients (e.g., mini-batches)
- Low memory overhead

**Limitations:**

- Requires gradient computation (automatic differentiation or numerical)
- Converges to local minima (no global search)
- Sensitive to learning rate choice
- May oscillate near optima

## Example

```python
import diffid as chron

optimiser = (
    diffid.Adam()
    .with_max_iter(5000)
    .with_step_size(0.001)
    .with_betas(0.9, 0.999)
    .with_threshold(1e-8)
)

result = optimiser.run(problem, initial_guess=[1.0, 2.0])
```

## Implementation Notes

- Supports automatic numerical gradient computation via central differences
- Bias correction is essential for early iterations
- Tracks both current position and best-found position

## References

1. Kingma, D.P. and Ba, J. (2015). "Adam: A Method for Stochastic Optimization". *ICLR 2015*. arXiv:1412.6980.
2. Reddi, S.J. et al. (2018). "On the Convergence of Adam and Beyond". *ICLR 2018*.

## See Also

- [API Reference](../../api-reference/python/optimizers.md#adam)
- [Choosing an Optimiser](../../guides/choosing-optimiser.md)
- [Tuning Optimisers](../../guides/tuning-optimizers.md)
