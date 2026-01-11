"""Contour plotting example using the ScalarBuilder API"""

from pathlib import Path

import chronopt as chron
import matplotlib.pyplot as plt
import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """Classic Rosenbrock banana function."""
    x = np.asarray(x, dtype=float)
    return float((1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


# Setup
builder = (
    chron.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", 1.0)
    .with_parameter("y", 1.0)
)
problem = builder.build()

# Optimise
optimiser = chron.NelderMead().with_max_iter(500).with_threshold(1e-8)
result = optimiser.run(problem, [-1.5, 1.5])

# Plot
contour_set = chron.plotting.contour(
    problem,
    x_bounds=(-2.0, 2.0),
    y_bounds=(-1.0, 3.0),
    grid_size=200,
    levels=np.logspace(-1, 3, 15),
    show=False,
)

ax = contour_set.axes
ax.plot(
    result.x[0],
    result.x[1],
    marker="*",
    color="red",
    markersize=12,
    label="Optimised x",
)
ax.legend()

print(f"Optimised parameters: {result.x}")
print(f"Objective value: {result.value:.3e}")

output_path = Path(__file__).with_suffix(".png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved contour plot to: {output_path}")
