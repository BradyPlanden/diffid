# 5-Minute Quickstart

This quickstart guide demonstrates scalar optimisation using the classic Rosenbrock function.

## The Problem

The Rosenbrock function is defined as:

$$f(x, y) = (1 - x)^2 + 100(y - x^2)^2$$

The global minimum is at $(x, y) = (1, 1)$ with $f(1, 1) = 0$.

## Basic Example

```python
import numpy as np
import chronopt as chron


def rosenbrock(x):
    """The Rosenbrock function - a classic optimisation test problem."""
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value])


# Build the problem
builder = (
    chron.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", 1.5)   # Initial guess
    .with_parameter("y", -1.5)  # Initial guess
)
problem = builder.build()

# Run optimisation (uses Nelder-Mead by default)
result = problem.optimise()

# Display results
print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")
print(f"Success: {result.success}")
print(f"Iterations: {result.iterations}")
```

**Output:**
```
Optimal parameters: [1.0, 1.0]
Objective value: 0.000e+00
Success: True
Iterations: 157
```

## Understanding the Code

1. **Define the objective function**: The function must accept a NumPy array and return a NumPy array

2. **Create a builder**: `ScalarBuilder()` is used for scalar optimisation problems where you directly evaluate a function

3. **Add parameters**: Use `with_parameter(name, initial_value)` to define decision variables

4. **Build the problem**: Call `build()` to create an optimisable problem instance

5. **Run optimisation**: Call `optimise()` to run the default optimiser (Nelder-Mead)

## Using Different Optimisers

You can specify which optimiser to use:

### CMA-ES

```python
# Use CMA-ES for global search
optimiser = chron.CMAES().with_max_iter(1000).with_step_size(0.5)
result = optimiser.run(problem, [1.5, -1.5])

print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")
```

### Adam (Gradient-Based)

```python
# Use Adam optimiser
optimiser = chron.Adam().with_max_iter(1000).with_step_size(0.01)
result = optimiser.run(problem, [1.5, -1.5])

print(f"Optimal parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")
```

## Visualising the Optimisation

If you installed the `plotting` extra, you can visualise the optimisation landscape:

```python
import numpy as np
import matplotlib.pyplot as plt
import chronopt as chron


def rosenbrock(x):
    value = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    return np.asarray([value])


# Create a grid for plotting
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = rosenbrock([X[i, j], Y[i, j]])[0]

# Plot contours
plt.figure(figsize=(10, 8))
levels = np.logspace(-1, 3.5, 20)
plt.contour(X, Y, Z, levels=levels, cmap='viridis')
plt.colorbar(label='f(x, y)')

# Mark the optimum
plt.plot(1.0, 1.0, 'r*', markersize=20, label='Global minimum')

# Run optimisation and plot path
builder = (
    chron.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", -1.5)
    .with_parameter("y", -0.5)
)
problem = builder.build()
result = problem.optimise()

plt.plot(result.x[0], result.x[1], 'go', markersize=10, label='Found optimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rosenbrock Function Optimisation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

![Rosenbrock contour plot](rosenbrock_contour.png)

## Next Steps

- **[First ODE Fit](first-ode-fit.md)**: Learn how to fit differential equations to data
- **[Core Concepts](concepts.md)**: Understand the builder pattern and problem types
- **[Choosing an Optimiser](../guides/choosing-optimiser.md)**: Learn when to use each optimiser
- **[Tutorials](../tutorials/index.md)**: Explore interactive Jupyter notebooks
