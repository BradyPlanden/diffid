use chronopt::prelude::*;
use chronopt::problem::builders_old::BuilderParameterExt;

/// Test evidence calculation against known analytical result.
/// For a Gaussian N(x | 0, σ²) with uniform prior U(a, b),
/// the evidence is Z = (1/(b-a)) * ∫ N(x | 0, σ²) dx ≈ (1/(b-a)) for wide priors.
#[test]
fn gaussian_evidence_accuracy() {
    let sigma: f64 = 1.0;
    let prior_lower: f64 = -10.0;
    let prior_upper: f64 = 10.0;

    // For a Gaussian with mean 0 and wide uniform prior, the evidence is approximately
    // Z ≈ 1/(b-a) when the prior is much wider than the likelihood
    let log_z_approx = -(prior_upper - prior_lower).ln();

    let problem = ScalarProblemBuilder::new()
        .with_objective(move |x: &[f64]| {
            let log_norm = sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            0.5 * (x[0] / sigma).powi(2) + log_norm
        })
        .with_parameter(ParameterSpec::new(
            "x",
            0.0,
            Some((prior_lower, prior_upper)),
        ))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(256)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-4)
        .with_seed(42);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(nested.draws() > 0, "should produce samples");
    assert!(
        nested.log_evidence().is_finite(),
        "evidence should be finite"
    );

    // Evidence should be within reasonable range, given number of samples
    let evidence_error = (nested.log_evidence() - log_z_approx).abs();
    assert!(
        evidence_error < 0.2,
        "evidence error too large: got {:.4}, expected ~{:.4}, error {:.4}",
        nested.log_evidence(),
        log_z_approx,
        evidence_error
    );
}

/// Test bimodal distribution to ensure both modes are sampled.
#[test]
fn bimodal_distribution_samples_both_modes() {
    let mu1 = -3.0;
    let mu2 = 3.0;
    let sigma: f64 = 0.5;

    let problem = ScalarProblemBuilder::new()
        .with_objective(move |x: &[f64]| {
            // Mixture of two Gaussians: -log(0.5 * N(μ1, σ²) + 0.5 * N(μ2, σ²))
            let log_p1 = -0.5 * ((x[0] - mu1) / sigma).powi(2);
            let log_p2 = -0.5 * ((x[0] - mu2) / sigma).powi(2);
            let max_log = log_p1.max(log_p2);
            let log_sum = max_log + ((log_p1 - max_log).exp() + (log_p2 - max_log).exp()).ln();
            // Return negative log likelihood
            -(log_sum - 2.0_f64.ln() - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln())
        })
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-10.0, 10.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(256)
        .with_expansion_factor(0.3)
        .with_termination_tolerance(1e-3)
        .with_seed(999);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(nested.draws() > 10, "should produce multiple samples");

    // Count samples near each mode
    let mut near_mode1 = 0;
    let mut near_mode2 = 0;

    for sample in nested.posterior() {
        let x = sample.position[0];
        if (x - mu1).abs() < 0.1 {
            near_mode1 += 1;
        }
        if (x - mu2).abs() < 0.1 {
            near_mode2 += 1;
        }
    }

    assert!(near_mode1 > 0, "should sample from first mode at x={}", mu1);
    assert!(
        near_mode2 > 0,
        "should sample from second mode at x={}",
        mu2
    );
}

/// Test that infinite likelihoods are handled gracefully.
#[test]
fn handles_infinite_likelihood() {
    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| {
            if x[0].abs() > 2.0 {
                f64::INFINITY
            } else {
                0.5 * x[0].powi(2)
            }
        })
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(32)
        .with_expansion_factor(0.2)
        .with_seed(456);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(
        nested.draws() > 0,
        "should produce samples despite infinite regions"
    );
    assert!(
        nested.log_evidence().is_finite(),
        "evidence should be finite"
    );

    // All samples should be in valid region
    for sample in nested.posterior() {
        assert!(
            sample.position[0].abs() <= 2.0,
            "sample at {} should be in valid region",
            sample.position[0]
        );
    }
}

/// Test high-dimensional problem scaling.
#[test]
fn high_dimensional_problem() {
    const DIM: usize = 8;

    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| 0.5 * x.iter().map(|xi| xi.powi(2)).sum::<f64>())
        .with_parameter(ParameterSpec::new("x0", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x1", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x2", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x3", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x4", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x5", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x6", 0.0, Some((-3.0, 3.0))))
        .with_parameter(ParameterSpec::new("x7", 0.0, Some((-3.0, 3.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(512)
        .with_expansion_factor(0.1)
        .with_termination_tolerance(1e-2)
        .with_seed(789);

    let nested = sampler.run(&problem, vec![0.0; DIM]);

    assert!(nested.draws() > 0, "should produce samples");
    assert_eq!(
        nested.mean().len(),
        DIM,
        "mean should have correct dimension"
    );

    for (i, &m) in nested.mean().iter().enumerate() {
        assert!(m.is_finite(), "mean[{}] should be finite", i);
        assert!(m.abs() < 1.0, "mean[{}] = {} should be near origin", i, m);
    }
}

/// Test very peaked (nearly delta function) posterior.
#[test]
fn sharp_peaked_posterior() {
    let sigma: f64 = 0.01;

    let problem = ScalarProblemBuilder::new()
        .with_objective(move |x: &[f64]| 0.5 * (x[0] / sigma).powi(2))
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-1.0, 1.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.05)
        .with_termination_tolerance(1e-3)
        .with_seed(321);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(nested.draws() > 0, "should produce samples");
    assert!(
        nested.log_evidence().is_finite(),
        "evidence should be finite"
    );
    assert!(
        nested.mean()[0].abs() < 0.1,
        "mean should be near zero, got {}",
        nested.mean()[0]
    );
}

/// Test with very small evidence (large negative log Z).
#[test]
fn very_small_evidence() {
    let offset: f64 = 100.0;

    let problem = ScalarProblemBuilder::new()
        .with_objective(move |x: &[f64]| offset + 0.5 * x[0].powi(2))
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(111);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(
        nested.log_evidence() < -50.0,
        "evidence should be very small"
    );
    assert!(
        nested.log_evidence().is_finite(),
        "evidence should be finite"
    );
    assert!(nested.draws() > 0, "should produce samples");
}

/// Test that posterior weights approximately sum to 1.
#[test]
fn posterior_weights_normalize() {
    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| 0.5 * x[0].powi(2))
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_seed(555);

    let nested = sampler.run(&problem, vec![0.0]);

    let log_z = nested.log_evidence();
    let mut weight_sum = 0.0;

    for sample in nested.posterior() {
        let log_posterior_weight = sample.log_weight + sample.log_likelihood - log_z;
        weight_sum += log_posterior_weight.exp();
    }

    assert!(
        (weight_sum - 1.0).abs() < 0.05,
        "weights should sum to ~1.0, got {}",
        weight_sum
    );
}

/// Test that mean matches weighted average.
#[test]
fn mean_matches_weighted_average() {
    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| 0.5 * (x[0] - 1.0).powi(2))
        .with_parameter(ParameterSpec::new("x", 1.0, Some((-3.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(128)
        .with_expansion_factor(0.2)
        .with_seed(777);

    let nested = sampler.run(&problem, vec![1.0]);

    let log_z = nested.log_evidence();
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for sample in nested.posterior() {
        let log_posterior_weight = sample.log_weight + sample.log_likelihood - log_z;
        let weight = log_posterior_weight.exp();
        weighted_sum += weight * sample.position[0];
        total_weight += weight;
    }

    let manual_mean = weighted_sum / total_weight;
    let reported_mean = nested.mean()[0];

    assert!(
        (manual_mean - reported_mean).abs() < 1e-6,
        "manual mean {} should match reported mean {}",
        manual_mean,
        reported_mean
    );
}

/// Test reproducibility with same seed.
#[test]
fn reproducibility_with_seed() {
    let make_problem = || {
        ScalarProblemBuilder::new()
            .with_objective(|x: &[f64]| 0.5 * x[0].powi(2))
            .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
            .build()
            .expect("failed to build problem")
    };

    let make_sampler = || {
        DynamicNestedSampler::new()
            .with_live_points(64)
            .with_expansion_factor(0.2)
            .with_seed(12345)
    };

    let problem1 = make_problem();
    let problem2 = make_problem();
    let sampler1 = make_sampler();
    let sampler2 = make_sampler();

    let nested1 = sampler1.run(&problem1, vec![0.0]);
    let nested2 = sampler2.run(&problem2, vec![0.0]);

    assert_eq!(nested1.draws(), nested2.draws(), "draws should match");
    assert_eq!(
        nested1.log_evidence(),
        nested2.log_evidence(),
        "evidence should match"
    );
    assert_eq!(nested1.mean(), nested2.mean(), "mean should match");
}

/// Test that all samples respect parameter bounds.
#[test]
fn respects_parameter_bounds() {
    let lower = -2.0;
    let upper = 3.0;

    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| 0.5 * x[0].powi(2))
        .with_parameter(ParameterSpec::new("x", 0.0, Some((lower, upper))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(444);

    let nested = sampler.run(&problem, vec![0.0]);

    for sample in nested.posterior() {
        let x = sample.position[0];
        assert!(
            x >= lower && x <= upper,
            "sample {} should be within bounds [{}, {}]",
            x,
            lower,
            upper
        );
    }
}

/// Test that information is non-negative.
#[test]
fn information_is_non_negative() {
    let problem = ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| 0.5 * x[0].powi(2))
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_seed(666);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(
        nested.information() >= 0.0,
        "information should be non-negative, got {}",
        nested.information()
    );
    assert!(
        nested.information().is_finite(),
        "information should be finite"
    );
}

/// Test that narrow posteriors have higher information than wide ones.
#[test]
fn information_increases_with_constraint() {
    let make_problem = |sigma: f64| {
        ScalarProblemBuilder::new()
            .with_objective(move |x: &[f64]| {
                let log_norm = sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
                0.5 * (x[0] / sigma).powi(2) + log_norm
            })
            .with_parameter(ParameterSpec::new("x", 0.0, Some((-10.0, 10.0))))
            .build()
            .expect("failed to build problem")
    };

    let make_sampler = |seed: u64| {
        DynamicNestedSampler::new()
            .with_live_points(128)
            .with_expansion_factor(0.2)
            .with_seed(seed)
    };

    let wide_problem = make_problem(2.0);
    let narrow_problem = make_problem(0.5);

    let wide_nested = make_sampler(100).run(&wide_problem, vec![0.0]);
    let narrow_nested = make_sampler(101).run(&narrow_problem, vec![0.0]);

    assert!(
        narrow_nested.information() > wide_nested.information(),
        "narrow posterior (info={}) should have higher information than wide (info={})",
        narrow_nested.information(),
        wide_nested.information()
    );
}

/// Test termination with high information content.
#[test]
fn terminates_with_high_information() {
    let sigma: f64 = 0.1;

    let problem = ScalarProblemBuilder::new()
        .with_objective(move |x: &[f64]| {
            let log_norm = sigma.ln() + 0.5 * (2.0 * std::f64::consts::PI).ln();
            0.5 * (x[0] / sigma).powi(2) + log_norm
        })
        .with_parameter(ParameterSpec::new("x", 0.0, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem");

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.1)
        .with_termination_tolerance(1e-3)
        .with_seed(888);

    let nested = sampler.run(&problem, vec![0.0]);

    assert!(nested.draws() > 0, "should produce samples");
    assert!(
        nested.draws() < 10000,
        "should terminate, got {} draws",
        nested.draws()
    );
    assert!(
        nested.information().is_finite(),
        "information should be finite"
    );
}
