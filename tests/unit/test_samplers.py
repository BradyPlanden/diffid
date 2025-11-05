import math

import chronopt as chron
import numpy as np


def quadratic_potential(x: np.ndarray) -> np.ndarray:
    value = 0.5 * (x[0] ** 2)
    return np.asarray([value], dtype=float)


def test_metropolis_hastings_runs_and_returns_samples():
    problem = (
        chron.PythonBuilder()
        .with_callable(quadratic_potential)
        .with_parameter("x", 1.0)
        .build()
    )

    sampler = (
        chron.samplers.MetropolisHastings()
        .with_num_chains(3)
        .with_num_steps(400)
        .with_step_size(0.4)
        .with_seed(123)
    )

    samples = sampler.run(problem, [1.5])

    assert samples.draws == 3 * 400
    assert len(samples.chains) == 3
    assert all(len(chain) == 401 for chain in samples.chains)

    flattened = []
    burn_in = 50
    for chain in samples.chains:
        flattened.extend(sample[0] for sample in chain[burn_in:])

    mean_estimate = float(np.mean(flattened))
    assert math.isfinite(mean_estimate)
    assert abs(mean_estimate) < 0.2
