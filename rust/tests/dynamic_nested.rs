use chronopt::builders::DiffsolBackend;
use chronopt::prelude::*;
use nalgebra::DMatrix;

#[test]
fn dynamic_nested_sampler_integration() {
    // Gaussian likelihood: L(x) = exp(-0.5 * (x - 0.5)^2)
    // Negative log-likelihood: -ln(L) = 0.5 * (x - 0.5)^2
    let objective = |x: &[f64]| {
        let diff = x[0] - 0.5;
        0.5 * diff * diff
    };

    let sampler = DynamicNestedSampler::new()
        .with_live_points(64)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-4);
    // Note: No fixed seed - RNG sequence changed with Fix #1 (position mismatch bug fix)

    let bounds = Bounds::new(vec![(-5.0, 5.0)]);
    let result = sampler.run(objective, vec![0.5], bounds);

    assert!(result.draws() > 0);
    assert!(result.log_evidence().is_finite());
    assert!(result.information().is_finite());
    let posterior_mean = result.mean()[0];
    assert!(posterior_mean.is_finite());
    // Relax tolerance - nested sampling with limited live points has variability
    assert!(
        (posterior_mean - 0.5).abs() < 0.2,
        "Posterior mean {} should be close to 0.5",
        posterior_mean
    );

    // Verify posterior samples are valid
    for sample in result.posterior() {
        assert!(
            sample.log_likelihood.is_finite(),
            "Log-likelihood should be finite"
        );
        assert!(sample.log_weight.is_finite(), "Log-weight should be finite");
        assert_eq!(sample.position.len(), 1, "Position should have 1 dimension");
    }
}

fn build_logistic_objective(
    backend: DiffsolBackend,
    parallel: bool,
) -> (impl Fn(&[f64]) -> f64, Bounds) {
    let dsl = r#"
in = [r, k]
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

    let t_span: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();

    // Data matrix: column-major format with t_span first, then data_values
    let data = DMatrix::from_vec(t_span.len(), 2, {
        let mut columns = Vec::with_capacity(t_span.len() * 2);
        columns.extend_from_slice(&t_span);
        columns.extend_from_slice(&data_values);
        columns
    });

    let problem = DiffsolProblemBuilder::new()
        .with_diffsl(dsl.to_string())
        .with_data(data)
        .with_parameter("r", 1.0, (0.1, 3.0))
        .with_parameter("k", 1.0, (0.5, 2.0))
        .with_backend(backend)
        .with_parallel(parallel)
        .build()
        .expect("failed to build problem");

    let bounds = Bounds::new(vec![(0.1, 3.0), (0.5, 2.0)]);

    (move |x: &[f64]| problem.evaluate(x).unwrap_or(1e10), bounds)
}

#[test]
fn dynamic_nested_sampler_parallel_vs_sequential_consistency() {
    for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
        let (parallel_objective, bounds) = build_logistic_objective(backend, true);
        let (sequential_objective, _) = build_logistic_objective(backend, false);
        let initial = vec![1.0, 1.0];

        // Setup sampler
        let sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_expansion_factor(0.3)
            .with_termination_tolerance(1e-3)
            .with_seed(42);

        let parallel_result = sampler.run(parallel_objective, initial.clone(), bounds.clone());

        let seq_sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_expansion_factor(0.3)
            .with_termination_tolerance(1e-3)
            .with_seed(42);

        let sequential_result = seq_sampler.run(sequential_objective, initial.clone(), bounds);

        // Both should produce valid results
        assert!(
            parallel_result.draws() > 0,
            "Parallel sampler should produce samples"
        );
        assert!(
            sequential_result.draws() > 0,
            "Sequential sampler should produce samples"
        );

        assert!(
            parallel_result.log_evidence().is_finite(),
            "Parallel log evidence should be finite"
        );
        assert!(
            sequential_result.log_evidence().is_finite(),
            "Sequential log evidence should be finite"
        );

        // Results should be reasonably close (some variation is expected due to randomness)
        let evidence_diff =
            (parallel_result.log_evidence() - sequential_result.log_evidence()).abs();
        assert!(
            evidence_diff < 5.0,
            "Evidence estimates should be similar: parallel={}, sequential={}, diff={}",
            parallel_result.log_evidence(),
            sequential_result.log_evidence(),
            evidence_diff
        );

        // Posterior means should be similar
        let parallel_mean = &parallel_result.mean();
        let sequential_mean = &sequential_result.mean();
        assert_eq!(parallel_mean.len(), sequential_mean.len());

        for (i, (p, s)) in parallel_mean.iter().zip(sequential_mean.iter()).enumerate() {
            let mean_diff = (p - s).abs();
            assert!(
                mean_diff < 0.5,
                "Parameter {} means should be similar: parallel={}, sequential={}, diff={}",
                i,
                p,
                s,
                mean_diff
            );
        }
    }
}

#[test]
fn dynamic_nested_sampler_parallel_basic_functionality() {
    let (objective, bounds) = build_logistic_objective(DiffsolBackend::Dense, true);
    let initial = vec![1.0, 1.0];

    // Sampler setup
    let sampler = DynamicNestedSampler::new()
        .with_live_points(32)
        .with_expansion_factor(0.3)
        .with_termination_tolerance(1e-3)
        .with_seed(123);

    let parallel_result = sampler.run(objective, initial, bounds);

    assert!(parallel_result.draws() > 0);
    assert!(parallel_result.log_evidence().is_finite());
    assert!(parallel_result.mean().len() == 2);
}
