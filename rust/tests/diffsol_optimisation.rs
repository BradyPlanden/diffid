use chronopt::prelude::*;
use chronopt::problem::ParameterSpec;
use nalgebra::DMatrix;

#[test]
fn diffsol_builder_supports_end_to_end_optimisation() {
    let dsl = r#"
in = [a]
a { 1 }
u_i { y = 0.1 }
F_i { a * y }
"#;

    let true_param = 1.2_f64;
    let y0 = 0.1_f64;

    let t_span: Vec<f64> = (0..20).map(|i| i as f64 * 0.05).collect();
    let data_values: Vec<f64> = t_span.iter().map(|t| y0 * (true_param * t).exp()).collect();

    let mut data = Vec::with_capacity(t_span.len() * 2);
    for (t, y) in t_span.iter().zip(data_values.iter()) {
        data.push(*t);
        data.push(*y);
    }

    let data_matrix = DMatrix::from_row_slice(t_span.len(), 2, &data);
    let params = ParameterSpec::new("a", 0.3, None);

    let optimiser = NelderMead::new()
        .with_max_iter(400)
        .with_threshold(1e-12)
        .with_sigma0(0.25);

    let builder = DiffsolBuilder::new()
        .with_diffsl(dsl.to_string())
        .with_data(data_matrix)
        .with_parameter(params)
        .with_rtol(1e-7)
        .with_atol(1e-7)
        .with_optimiser(optimiser.clone());

    let problem = builder.build().expect("problem should build");

    let true_cost = problem
        .evaluate(&[true_param])
        .expect("evaluation at true parameter should succeed");
    assert!(
        true_cost < 1e-10,
        "expected near-zero cost at true parameter"
    );

    let initial_guess = 0.5;
    let initial_cost = problem
        .evaluate(&[initial_guess])
        .expect("evaluation at initial guess should succeed");
    assert!(initial_cost > true_cost);

    let result = problem.optimize(Some(vec![initial_guess]), None);

    assert!(
        result.success,
        "optimisation should succeed: {}",
        result.message
    );
    assert!(
        result.fun < initial_cost,
        "optimisation should reduce cost ({} -> {})",
        initial_cost,
        result.fun
    );
    assert!(
        (result.x[0] - true_param).abs() < 1e-2,
        "expected parameter close to true value ({} vs {})",
        result.x[0],
        true_param
    );

    // Ensure the configured default optimiser was used by clearing builder defaults
    let builder_without_default = DiffsolBuilder::new()
        .with_diffsl(dsl.to_string())
        .with_data(DMatrix::from_row_slice(t_span.len(), 2, &data))
        .with_parameter(ParameterSpec::new("a", true_param, None));

    let problem_without_default = builder_without_default
        .build()
        .expect("problem should build without default optimiser");

    let explicit_result = optimiser.run(&problem_without_default, vec![initial_guess]);

    assert!(
        explicit_result.success,
        "explicit optimiser run should succeed: {}",
        explicit_result.message
    );
    assert!(
        explicit_result.fun < initial_cost,
        "explicit optimiser should reduce cost"
    );
}
