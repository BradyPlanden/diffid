"""Comprehensive unit tests for the dynamic nested sampler."""

from __future__ import annotations

import math

import chronopt as chron
import numpy as np
import pytest
from scipy.stats import norm


def test_gaussian_evidence_accuracy():
    """Test evidence calculation against known analytical result.

    For a Gaussian likelihood N(x | μ, σ²) with uniform prior U(a, b),
    the evidence is: Z = [Φ((b-μ)/σ) - Φ((a-μ)/σ)] / (b-a) where Φ is the
    standard normal CDF.
    """
    mu = 0.0
    sigma = 10.0
    prior_lower = -25.0
    prior_upper = 25.0

    # Analytical log evidence
    prior_width = prior_upper - prior_lower
    z_upper = (prior_upper - mu) / sigma
    z_lower = (prior_lower - mu) / sigma
    cdf_diff = norm.cdf(z_upper) - norm.cdf(z_lower)

    log_z_analytical = np.log(cdf_diff) - np.log(prior_width)

    def gaussian_nll(x: list[float]) -> float:
        """Negative log likelihood for Gaussian."""
        diff = x[0] - mu
        return 0.5 * (diff / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)

    problem = (
        chron.ScalarBuilder()
        .with_objective(gaussian_nll)
        .with_parameter("x", mu, bounds=(prior_lower, prior_upper))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-5)
        .with_seed(42)
    )

    nested = sampler.run(problem)

    # Evidence should be within a reasonable tolerance of the true value
    assert abs(nested.log_evidence - log_z_analytical) < 0.1, (
        f"Evidence error too large: got {nested.log_evidence:.4f}, "
        f"expected {log_z_analytical:.4f}"
    )

    # Mean should be close to true mean
    assert abs(nested.mean[0] - mu) < 0.75

    # Information should be positive and finite
    assert nested.information > 0
    assert math.isfinite(nested.information)


def test_exponential_distribution_evidence():
    """Test with exponential distribution: p(x) ∝ exp(-λx) for x > 0.

    With uniform prior on [0, x_max], evidence is:
    Z = (1/x_max) * (1 - exp(-λ * x_max)) / λ
    """
    lambda_param = 2.0
    x_max = 7.5

    # Analytical log evidence
    numerator = 1.0 - np.exp(-lambda_param * x_max)
    log_z_analytical = np.log(numerator) - np.log(lambda_param) - np.log(x_max)

    def exponential_nll(x: list[float]) -> float:
        """Negative log likelihood for exponential."""
        return lambda_param * x[0]

    problem = (
        chron.ScalarBuilder()
        .with_objective(exponential_nll)
        .with_parameter("x", 1.0, bounds=(0.0, x_max))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.01)
        .with_termination_tolerance(1e-6)
        .with_seed(123)
    )

    nested = sampler.run(problem)

    # Evidence should be reasonably close
    assert abs(nested.log_evidence - log_z_analytical) < 0.1, (
        f"Evidence error: got {nested.log_evidence:.4f}, "
        f"expected {log_z_analytical:.4f}"
    )


def test_bimodal_gaussian_mixture():
    """Test sampler on well-separated bimodal distribution."""
    mu1, mu2 = -3.0, 3.0
    sigma = 0.5

    def bimodal_nll(x: list[float]) -> float:
        """Negative log likelihood for mixture of two Gaussians."""
        # Log of mixture: log(0.5 * N(μ1, σ²) + 0.5 * N(μ2, σ²))
        log_p1 = -0.5 * ((x[0] - mu1) / sigma) ** 2
        log_p2 = -0.5 * ((x[0] - mu2) / sigma) ** 2
        max_log = max(log_p1, log_p2)
        log_sum = max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p2 - max_log))
        # Add normalization and mixture weight
        return -(log_sum - np.log(2) - np.log(sigma) - 0.5 * np.log(2 * np.pi))

    problem = (
        chron.ScalarBuilder()
        .with_objective(bimodal_nll)
        .with_parameter("x", 0.0, bounds=(-10.0, 10.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-4)
        .with_seed(999)
    )

    nested = sampler.run(problem)

    # Check that samples exist near both modes
    positions = np.array([entry[0][0] for entry in nested.posterior])

    # Should have samples near both modes
    near_mode1 = np.sum(np.abs(positions - mu1) < 1.0)
    near_mode2 = np.sum(np.abs(positions - mu2) < 1.0)

    assert near_mode1 > 0, "Should sample from first mode"
    assert near_mode2 > 0, "Should sample from second mode"

    # Mean should be somewhere between the modes
    assert -4.0 < nested.mean[0] < 4.0


def test_handles_infinite_likelihood_gracefully():
    """Test that infinite likelihoods are filtered out."""

    def pathological_nll(x: list[float]) -> float:
        """Returns inf for some regions."""
        if abs(x[0]) > 2.0:
            return float("inf")
        return 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(pathological_nll)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(32)
        .with_expansion_factor(0.2)
        .with_seed(456)
    )

    nested = sampler.run(problem)

    # Should still produce valid results
    assert nested.draws > 0
    assert math.isfinite(nested.log_evidence)

    # All samples should be in valid region
    positions = np.array([entry[0][0] for entry in nested.posterior])
    assert np.all(np.abs(positions) <= 2.0)


def test_very_high_dimensional_problem():
    """Test scaling to higher dimensions."""
    dimension = 10

    def high_dim_quadratic(x: list[float]) -> float:
        """Simple quadratic in high dimensions."""
        return 0.5 * sum(xi**2 for xi in x)

    problem = chron.ScalarBuilder().with_objective(high_dim_quadratic)

    for i in range(dimension):
        problem = problem.with_parameter(f"x{i}", 0.0, bounds=(-3.0, 3.0))

    problem = problem.build()

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.15)
        .with_termination_tolerance(1e-4)
        .with_seed(789)
    )

    nested = sampler.run(problem)

    assert nested.draws > 0
    assert len(nested.mean) == dimension
    assert all(math.isfinite(m) for m in nested.mean)
    assert all(abs(m) < 1.0 for m in nested.mean)  # Should be near origin


def test_degenerate_posterior_delta_function():
    """Test with very peaked (nearly delta function) posterior."""

    def sharp_peak(x: list[float]) -> float:
        """Very narrow Gaussian."""
        sigma = 0.01
        return 0.5 * (x[0] / sigma) ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(sharp_peak)
        .with_parameter("x", 0.0, bounds=(-1.0, 1.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.05)  # Small expansion for narrow peak
        .with_termination_tolerance(1e-3)
        .with_seed(321)
    )

    nested = sampler.run(problem)

    assert nested.draws > 0
    assert math.isfinite(nested.log_evidence)
    # Mean should be very close to zero
    assert abs(nested.mean[0]) < 0.1


def test_very_small_evidence():
    """Test with problem that has very small evidence (large negative log Z)."""

    def large_offset(x: list[float]) -> float:
        """Likelihood with large constant offset."""
        return 100.0 + 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(large_offset)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(111)
    )

    nested = sampler.run(problem)

    # Evidence should be very negative but finite
    assert nested.log_evidence < -50
    assert math.isfinite(nested.log_evidence)
    assert nested.draws > 0


def test_posterior_weights_normalize():
    """Verify that posterior weights sum to approximately 1."""

    def simple_quadratic(x: list[float]) -> float:
        return 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(simple_quadratic)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_seed(555)
    )

    nested = sampler.run(problem)

    # Extract weights and normalize
    log_weights = np.array([entry[2] for entry in nested.posterior])
    log_likelihoods = np.array([entry[1] for entry in nested.posterior])

    # Posterior weights: w_i = exp(log_weight_i + log_likelihood_i - log_Z)
    log_posterior_weights = log_weights + log_likelihoods - nested.log_evidence
    posterior_weights = np.exp(log_posterior_weights)

    # Should sum to approximately 1
    weight_sum = np.sum(posterior_weights)
    assert abs(weight_sum - 1.0) < 0.05, f"Weights sum to {weight_sum}, expected ~1.0"


def test_mean_matches_weighted_average():
    """Verify that reported mean matches manual weighted average calculation."""

    def quadratic(x: list[float]) -> float:
        return 0.5 * (x[0] - 1.0) ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(quadratic)
        .with_parameter("x", 1.0, bounds=(-3.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_seed(777)
    )

    nested = sampler.run(problem)

    # Manual calculation
    positions = np.array([entry[0] for entry in nested.posterior])
    log_weights = np.array([entry[2] for entry in nested.posterior])
    log_likelihoods = np.array([entry[1] for entry in nested.posterior])

    log_posterior_weights = log_weights + log_likelihoods - nested.log_evidence
    posterior_weights = np.exp(log_posterior_weights)
    posterior_weights /= posterior_weights.sum()

    manual_mean = np.sum(posterior_weights[:, None] * positions, axis=0)

    # Should match reported mean
    np.testing.assert_allclose(nested.mean, manual_mean, rtol=1e-6, atol=1e-6)


def test_termination_with_high_information():
    """Ensure sampler terminates even with high information content."""

    def narrow_gaussian(x: list[float]) -> float:
        """Narrow Gaussian has high information."""
        sigma = 0.1
        return 0.5 * (x[0] / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)

    problem = (
        chron.ScalarBuilder()
        .with_objective(narrow_gaussian)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.1)
        .with_termination_tolerance(1e-3)
        .with_seed(888)
    )

    nested = sampler.run(problem)

    # Should terminate and produce results
    assert nested.draws > 0
    assert nested.draws < 10000  # Shouldn't run forever
    assert math.isfinite(nested.information)


def test_reproducibility_with_seed():
    """Test that same seed produces identical results."""

    def simple_problem(x: list[float]) -> float:
        return 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(simple_problem)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler1 = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(12345)
    )

    sampler2 = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(12345)
    )

    nested1 = sampler1.run(problem)
    nested2 = sampler2.run(problem)

    # Should produce identical results
    assert nested1.draws == nested2.draws
    assert nested1.log_evidence == nested2.log_evidence
    np.testing.assert_array_equal(nested1.mean, nested2.mean)


def test_live_points_adapt_with_information():
    """Test that live points increase with information content."""
    # This is more of an observational test - we can't easily assert
    # exact behavior, but we can verify the sampler runs

    def multimodal(x: list[float]) -> float:
        """Multimodal has higher information."""
        return -np.log(
            np.exp(-0.5 * ((x[0] + 2) / 0.3) ** 2)
            + np.exp(-0.5 * ((x[0] - 2) / 0.3) ** 2)
        )

    problem = (
        chron.ScalarBuilder()
        .with_objective(multimodal)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(32)
        .with_expansion_factor(0.5)  # Allow significant expansion
        .with_seed(999)
    )

    nested = sampler.run(problem)

    # Just verify it completes successfully
    assert nested.draws > 0
    assert math.isfinite(nested.information)


def test_respects_parameter_bounds():
    """Verify all samples respect parameter bounds."""
    lower, upper = -2.0, 3.0

    def bounded_quadratic(x: list[float]) -> float:
        return 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(bounded_quadratic)
        .with_parameter("x", 0.0, bounds=(lower, upper))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(444)
    )

    nested = sampler.run(problem)

    # All samples should be within bounds
    positions = np.array([entry[0][0] for entry in nested.posterior])
    assert np.all(positions >= lower)
    assert np.all(positions <= upper)


def test_information_is_non_negative():
    """Information (KL divergence) should always be non-negative."""

    def simple_problem(x: list[float]) -> float:
        return 0.5 * x[0] ** 2

    problem = (
        chron.ScalarBuilder()
        .with_objective(simple_problem)
        .with_parameter("x", 0.0, bounds=(-5.0, 5.0))
        .build()
    )

    sampler = (
        chron.DynamicNestedSampler()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(666)
    )

    nested = sampler.run(problem)

    assert nested.information >= 0.0
    assert math.isfinite(nested.information)


def test_information_increases_with_constraint():
    """More constrained posteriors should have higher information."""

    def make_gaussian_problem(sigma: float):
        def nll(x: list[float]) -> float:
            return 0.5 * (x[0] / sigma) ** 2 + np.log(sigma) + 0.5 * np.log(2 * np.pi)

        return (
            chron.ScalarBuilder()
            .with_objective(nll)
            .with_parameter("x", 0.0, bounds=(-10.0, 10.0))
            .build()
        )

    def sampler_config(seed):
        return (
            chron.DynamicNestedSampler()
            .with_live_points(128)
            .with_expansion_factor(0.2)
            .with_seed(seed)
        )

    # Wide posterior (low information)
    wide_problem = make_gaussian_problem(sigma=2.0)
    wide_nested = sampler_config(100).run(wide_problem)

    # Narrow posterior (high information)
    narrow_problem = make_gaussian_problem(sigma=0.5)
    narrow_nested = sampler_config(101).run(narrow_problem)

    # Narrow posterior should have higher information
    assert narrow_nested.information > wide_nested.information


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
