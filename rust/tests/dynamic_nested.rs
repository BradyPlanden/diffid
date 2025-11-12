use chronopt::prelude::*;
use chronopt::problem::builders::BuilderParameterExt;

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
        .with_seed(1337);

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
