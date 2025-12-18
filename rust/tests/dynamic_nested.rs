use chronopt::prelude::*;
use chronopt::problem::builders_old::BuilderParameterExt;
use chronopt::problem::DiffsolBackend;
use nalgebra::DMatrix;

fn gaussian_problem() -> Problem {
    ScalarProblemBuilder::new()
        .with_objective(|x: &[f64]| {
            let diff = x[0] - 0.5;
            0.5 * diff * diff
        })
        .with_parameter(ParameterSpec::new("x", 0.6, Some((-5.0, 5.0))))
        .build()
        .expect("failed to build problem")
}

#[test]
fn dynamic_nested_sampler_integration() {
    let problem = gaussian_problem();
    let sampler = DynamicNestedSampler::new()
        .with_live_points(32)
        .with_expansion_factor(0.2)
        .with_termination_tolerance(1e-4)
        .with_seed(37);

    let result = sampler.run_nested(&problem, vec![0.5]);

    assert!(result.draws() > 0);
    assert!(result.log_evidence().is_finite());
    assert!(result.information().is_finite());
    let posterior_mean = result.mean()[0];
    assert!(posterior_mean.is_finite());
    assert!((posterior_mean - 0.5).abs() < 0.05);

    let mut prev = f64::NEG_INFINITY;
    for sample in result.posterior() {
        assert!(sample.log_likelihood.is_finite());
        assert!(sample.log_weight.is_finite());
        assert_eq!(sample.position.len(), 1);
        assert!(sample.log_likelihood >= prev || sample.log_weight <= prev);
        prev = sample.log_likelihood;
    }
}

fn build_logistic_problem(backend: DiffsolBackend, parallel: bool) -> Problem {
    let dsl = r#"
in = [r, k]
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

    let t_span: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();
    let data = DMatrix::from_fn(t_span.len(), 2, |i, j| match j {
        0 => t_span[i],
        1 => data_values[i],
        _ => unreachable!(),
    });

    DiffsolProblemBuilder::new()
        .with_diffsl(dsl.to_string())
        .with_data(data)
        .with_parameter(ParameterSpec::new("r", 1.0, Some((0.1, 3.0))))
        .with_parameter(ParameterSpec::new("k", 1.0, Some((0.5, 2.0))))
        .with_backend(backend)
        .with_parallel(parallel)
        .build()
        .expect("failed to build problem")
}

#[test]
fn dynamic_nested_sampler_parallel_vs_sequential_consistency() {
    for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
        let parallel_problem = build_logistic_problem(backend, true);
        let sequential_problem = build_logistic_problem(backend, false);
        let initial = vec![1.0, 1.0];

        // Setup sampler
        let sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_expansion_factor(0.3)
            .with_termination_tolerance(1e-3)
            .with_seed(42);

        let parallel_result = { sampler.run_nested(&parallel_problem, initial.clone()) };

        let seq_sampler = DynamicNestedSampler::new()
            .with_live_points(32)
            .with_expansion_factor(0.3)
            .with_termination_tolerance(1e-3)
            .with_seed(42);

        let sequential_result = { seq_sampler.run_nested(&sequential_problem, initial.clone()) };

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
    let problem = build_logistic_problem(DiffsolBackend::Dense, true);
    let initial = vec![1.0, 1.0];

    // Sampler setup
    let sampler = DynamicNestedSampler::new()
        .with_live_points(32)
        .with_expansion_factor(0.3)
        .with_termination_tolerance(1e-3)
        .with_seed(123);

    let parallel_result = sampler.run_nested(&problem, initial.clone());

    assert!(parallel_result.draws() > 0);
    assert!(parallel_result.log_evidence().is_finite());
    assert!(parallel_result.mean().len() == 2);
}
