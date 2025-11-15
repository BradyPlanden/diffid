import chronopt as chron
import numpy as np


def _logistic_dsl() -> str:
    return """
in = [r, k]
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"""


def _logistic_data(n: int = 40) -> np.ndarray:
    t_span = np.linspace(0.0, 4.0, n)
    y = 0.1 * np.exp(t_span)
    return np.column_stack((t_span, y))


def _build_diffsol_problem(parallel: bool) -> chron.Problem:
    data = _logistic_data(40)
    builder = (
        chron.DiffsolBuilder()
        .with_diffsl(_logistic_dsl())
        .with_data(data)
        .with_parameter("r", 1.0, bounds=(0.1, 3.0))
        .with_parameter("k", 1.0, bounds=(0.5, 2.0))
        .with_parallel(parallel)
    )
    return builder.build()


def test_dynamic_nested_diffsol_parallel_vs_sequential():
    # Test parallel problem execution
    parallel_problem = _build_diffsol_problem(parallel=True)
    # Test sequential problem execution
    sequential_problem = _build_diffsol_problem(parallel=False)

    initial = [1.0, 1.0]

    sampler = (
        chron.sampler.DynamicNestedSampler()
        .with_live_points(32)
        .with_expansion_factor(0.3)
        .with_termination_tolerance(1e-3)
        .with_seed(42)
    )

    parallel_result = sampler.run(parallel_problem, initial)
    sequential_result = sampler.run(sequential_problem, initial)

    assert parallel_result.draws > 0
    assert sequential_result.draws > 0
    assert np.isfinite(parallel_result.log_evidence)
    assert np.isfinite(sequential_result.log_evidence)

    evidence_diff = abs(parallel_result.log_evidence - sequential_result.log_evidence)
    assert evidence_diff < 5.0

    assert len(parallel_result.mean) == len(sequential_result.mean) == 2
    mean_diffs = np.abs(
        np.array(parallel_result.mean) - np.array(sequential_result.mean)
    )
    assert np.all(mean_diffs < 0.5)


def test_dynamic_nested_sampler_parallel_fallback_for_non_parallel_problems():
    # Scalar problem does not support parallel evaluation; sampler should still work
    problem = (
        chron.ScalarBuilder()
        .with_callable(lambda x: 0.5 * (x[0] - 0.5) ** 2)
        .with_parameter("x", 0.5, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.sampler.DynamicNestedSampler()
        .with_live_points(24)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-3)
        .with_seed(7)
    )

    result = sampler.run(problem, [0.5])

    assert result.draws > 0
    assert np.isfinite(result.log_evidence)
    assert np.isfinite(result.information)
