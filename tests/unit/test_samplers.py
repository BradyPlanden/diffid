import math

import chronopt as chron
import numpy as np
import pytest


def quadratic_potential(x: np.ndarray) -> np.ndarray:
    value = 0.5 * (x[0] ** 2)
    return np.asarray([value], dtype=float)


def test_metropolis_hastings_runs_and_returns_samples():
    problem = (
        chron.ScalarBuilder()
        .with_callable(quadratic_potential)
        .with_parameter("x", 1.0)
        .build()
    )

    sampler = (
        chron.MetropolisHastings()
        .with_num_chains(3)
        .with_iterations(400)
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


def test_dynamic_nested_sampler_runs_on_scalar_problem():
    problem = (
        chron.ScalarBuilder()
        .with_callable(quadratic_potential)
        .with_parameter("x", 0.5)
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(32)
        .with_expansion_factor(0.1)
        .with_seed(99)
    )

    nested = sampler.run(problem)

    assert nested.draws > 0
    assert math.isfinite(nested.log_evidence)
    assert math.isfinite(nested.information)
    assert nested.mean == pytest.approx([0.0], abs=0.3)


def test_dynamic_nested_invalid_live_points_are_clamped():
    problem = (
        chron.ScalarBuilder()
        .with_callable(quadratic_potential)
        .with_parameter("x", 1.0)
        .build()
    )

    sampler = chron.DynamicNestedSampler().with_live_points(1)
    nested = sampler.run(problem)

    assert nested.draws >= 0


def test_dynamic_nested_requires_problem_instance():
    sampler = chron.DynamicNestedSampler()

    with pytest.raises(TypeError):
        sampler.run(object())  # type: ignore[arg-type]
