"""Model evidence example using the dynamic nested sampler."""

from __future__ import annotations

import diffid


def rosenbrock(x: list[float]) -> float:
    """Two-dimensional Rosenbrock objective."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


builder = (
    diffid.ScalarBuilder()
    .with_objective(rosenbrock)
    .with_parameter("x", initial_value=1.2)
    .with_parameter("y", initial_value=1.4)
    .with_optimiser(diffid.NelderMead().with_max_iter(2000))
)
problem = builder.build()

optimised = problem.optimise()

sampler = diffid.DynamicNestedSampler().with_live_points(256).with_seed(1234)
samples = sampler.run(problem, initial=optimised.x)

print("time       :", samples.time)
print("log(Z)      :", samples.log_evidence)
print("information :", samples.information)
print("evaluations :", samples.draws)
