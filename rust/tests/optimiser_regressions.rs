use diffid::optimisers::{Adam, NelderMead, TerminationReason, CMAES};
use diffid::prelude::{
    AskResult, Bounds, DiffsolProblemBuilder, ScalarProblemBuilder, SingleAskResult, Unbounded,
    VectorProblemBuilder,
};
use diffid::problem::ProblemError;
use nalgebra::DMatrix;

#[test]
fn scalar_builder_with_cmaes_is_used_by_problem_optimise() {
    let configured_optimiser = CMAES::new().with_max_iter(4).with_seed(7);

    let problem = ScalarProblemBuilder::new()
        .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
        .with_parameter("x0", 4.0, Unbounded)
        .with_parameter("x1", -3.0, Unbounded)
        .with_optimiser(configured_optimiser.clone())
        .build()
        .expect("problem should build");

    let implicit_result = problem.optimise(None, None);

    assert!(
        implicit_result.covariance.is_some(),
        "expected covariance when configured CMA-ES is used"
    );

    // Sanity check that explicit CMA-ES does produce covariance output.
    let explicit_cmaes_result = configured_optimiser.run(
        |x| problem.evaluate(x),
        vec![4.0, -3.0],
        Bounds::unbounded(2),
    );
    assert!(
        explicit_cmaes_result.covariance.is_some(),
        "explicit CMA-ES run should include covariance"
    );
}

#[test]
fn vector_builder_with_cmaes_is_used_by_problem_optimise() {
    let problem = VectorProblemBuilder::new()
        .with_function(
            |params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                let scale = params[0];
                Ok(vec![scale, 2.0 * scale, 3.0 * scale])
            },
        )
        .with_data(vec![1.0, 2.0, 3.0])
        .with_parameter("scale", 0.5, Unbounded)
        .with_optimiser(CMAES::new().with_max_iter(3).with_seed(7))
        .build()
        .expect("problem should build");

    let result = problem.optimise(None, None);
    assert!(
        result.covariance.is_some(),
        "expected covariance when configured CMA-ES is used"
    );
}

#[test]
fn nelder_mead_eval_error_terminates_with_function_evaluation_failed() {
    let optimiser = NelderMead::new().with_max_iter(10);

    let result = optimiser.run(
        |_x| -> Result<f64, std::io::Error> { Err(std::io::Error::other("boom")) },
        vec![0.0],
        Bounds::unbounded(1),
    );

    assert!(matches!(
        result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));
}

#[test]
fn adam_eval_error_terminates_with_function_evaluation_failed() {
    let optimiser = Adam::new().with_max_iter(10);

    let result = optimiser.run(
        |_x| -> Result<(f64, Vec<f64>), std::io::Error> { Err(std::io::Error::other("boom")) },
        vec![0.0],
        Bounds::unbounded(1),
    );

    assert!(matches!(
        result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));
}

#[test]
fn cmaes_eval_error_terminates_with_function_evaluation_failed() {
    let optimiser = CMAES::new().with_max_iter(10).with_seed(7);

    let result = optimiser.run(
        |_x| -> Result<f64, std::io::Error> { Err(std::io::Error::other("boom")) },
        vec![0.0],
        Bounds::unbounded(1),
    );

    assert!(matches!(
        result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));
    assert!(!result.success);
}

#[test]
fn cmaes_run_batch_eval_error_terminates_with_function_evaluation_failed() {
    let optimiser = CMAES::new()
        .with_max_iter(10)
        .with_population_size(4)
        .with_seed(7);

    let result = optimiser.run_batch(
        |xs| {
            xs.iter()
                .map(|_| Err::<f64, std::io::Error>(std::io::Error::other("boom")))
                .collect::<Vec<_>>()
        },
        vec![0.0],
        Bounds::unbounded(1),
    );

    assert!(matches!(
        result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));
    assert!(!result.success);
}

#[test]
fn diffsol_objective_error_semantics_are_consistent_across_optimisers() {
    let dsl = r"
in_i { a = 1 }
u_i { y = 0.1 }
F_i { a * y }
";

    let data = DMatrix::from_row_slice(
        4,
        2,
        &[
            0.0, 0.10, //
            0.2, 0.12, //
            0.1, 0.11, //
            0.3, 0.13, //
        ],
    );

    let problem = DiffsolProblemBuilder::new()
        .with_diffsl(dsl.to_string())
        .with_data(data)
        .with_parameter("a", 1.0, Unbounded)
        .build()
        .expect("diffsol problem should build");

    let nelder_mead_result = NelderMead::new().with_max_iter(10).run(
        |x| problem.evaluate(x),
        vec![1.0],
        Bounds::unbounded(1),
    );
    assert!(matches!(
        nelder_mead_result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));

    let cmaes_result = CMAES::new().with_max_iter(10).with_seed(7).run(
        |x| problem.evaluate(x),
        vec![1.0],
        Bounds::unbounded(1),
    );
    assert!(matches!(
        cmaes_result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));

    let adam_result = Adam::new().with_max_iter(10).run(
        |x| -> Result<(f64, Vec<f64>), ProblemError> {
            let value = problem.evaluate(x)?;
            Ok((value, vec![0.0]))
        },
        vec![1.0],
        Bounds::unbounded(1),
    );
    assert!(matches!(
        adam_result.termination,
        TerminationReason::FunctionEvaluationFailed(_)
    ));
}

#[test]
fn adam_run_with_bounds_dimension_mismatch_fails_fast() {
    let result = Adam::new().run(
        |x| {
            (
                x.iter().map(|v| v * v).sum::<f64>(),
                vec![2.0 * x[0], 2.0 * x[1]],
            )
        },
        vec![10.0, 10.0],
        Bounds::new(vec![(-1.0, 1.0)]),
    );

    match result.termination {
        TerminationReason::FunctionEvaluationFailed(msg) => {
            assert!(msg.contains("bounds dimension mismatch"));
        }
        other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
    }
}

#[test]
fn cmaes_run_with_bounds_dimension_mismatch_fails_fast() {
    let result = CMAES::new().with_seed(7).run(
        |x| x.iter().map(|v| v * v).sum::<f64>(),
        vec![10.0, 10.0],
        Bounds::new(vec![(-1.0, 1.0)]),
    );

    match result.termination {
        TerminationReason::FunctionEvaluationFailed(msg) => {
            assert!(msg.contains("bounds dimension mismatch"));
        }
        other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
    }
}

#[test]
fn nelder_mead_run_with_bounds_dimension_mismatch_fails_fast() {
    let result = NelderMead::new().run(
        |x| x.iter().map(|v| v * v).sum::<f64>(),
        vec![10.0, 10.0],
        Bounds::new(vec![(-1.0, 1.0)]),
    );

    match result.termination {
        TerminationReason::FunctionEvaluationFailed(msg) => {
            assert!(msg.contains("bounds dimension mismatch"));
        }
        other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
    }
}

#[test]
fn adam_init_with_bounds_dimension_mismatch_starts_terminated() {
    let (state, _first_point) = Adam::new().init(vec![10.0, 10.0], Bounds::new(vec![(-1.0, 1.0)]));
    match state.ask() {
        AskResult::Done(results) => match results.termination {
            TerminationReason::FunctionEvaluationFailed(msg) => {
                assert!(msg.contains("bounds dimension mismatch"));
            }
            other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
        },
        AskResult::Evaluate(_) => panic!("expected terminated state for mismatched bounds"),
    }
}

#[test]
fn cmaes_init_with_bounds_dimension_mismatch_starts_terminated() {
    let (state, _first_point) = CMAES::new().init(vec![10.0, 10.0], Bounds::new(vec![(-1.0, 1.0)]));
    match state.ask() {
        AskResult::Done(results) => match results.termination {
            TerminationReason::FunctionEvaluationFailed(msg) => {
                assert!(msg.contains("bounds dimension mismatch"));
            }
            other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
        },
        AskResult::Evaluate(_) => panic!("expected terminated state for mismatched bounds"),
    }
}

#[test]
fn nelder_mead_init_with_bounds_dimension_mismatch_starts_terminated() {
    let (state, _first_points) =
        NelderMead::new().init(vec![10.0, 10.0], Bounds::new(vec![(-1.0, 1.0)]));
    match state.ask() {
        AskResult::Done(results) => match results.termination {
            TerminationReason::FunctionEvaluationFailed(msg) => {
                assert!(msg.contains("bounds dimension mismatch"));
            }
            other => panic!("expected FunctionEvaluationFailed, got {other:?}"),
        },
        AskResult::Evaluate(_) => panic!("expected terminated state for mismatched bounds"),
    }
}

#[test]
fn adam_ask_single_matches_batch_ask_shape() {
    let (state, first_point) = Adam::new().init(vec![1.0, -1.0], Bounds::unbounded(2));

    match state.ask_single() {
        SingleAskResult::Evaluate(point) => assert_eq!(point, first_point),
        SingleAskResult::Done(_) => panic!("expected Evaluate from ask_single"),
    }

    match state.ask() {
        AskResult::Evaluate(points) => {
            assert_eq!(points.len(), 1);
            assert_eq!(points[0], first_point);
        }
        AskResult::Done(_) => panic!("expected Evaluate from ask"),
    }
}

#[test]
fn nelder_mead_ask_single_matches_batch_ask_shape() {
    let (state, first_points) = NelderMead::new().init(vec![2.0, 1.0], Bounds::unbounded(2));
    let first_point = first_points[0].clone();

    match state.ask_single() {
        SingleAskResult::Evaluate(point) => assert_eq!(point, first_point),
        SingleAskResult::Done(_) => panic!("expected Evaluate from ask_single"),
    }

    match state.ask() {
        AskResult::Evaluate(points) => {
            assert_eq!(points.len(), 1);
            assert_eq!(points[0], first_point);
        }
        AskResult::Done(_) => panic!("expected Evaluate from ask"),
    }
}

#[test]
fn adam_history_capture_is_opt_in_and_off_by_default() {
    let without_history = Adam::new().with_max_iter(8).with_threshold(0.0).run(
        |x| (x[0] * x[0], vec![2.0 * x[0]]),
        vec![1.0],
        Bounds::unbounded(1),
    );

    let with_history = Adam::new()
        .with_max_iter(8)
        .with_threshold(0.0)
        .with_history(true)
        .run(
            |x| (x[0] * x[0], vec![2.0 * x[0]]),
            vec![1.0],
            Bounds::unbounded(1),
        );

    assert_eq!(
        without_history.final_simplex.len(),
        1,
        "history should be disabled by default"
    );
    assert!(
        with_history.final_simplex.len() > 1,
        "history-enabled mode should retain multiple evaluated points"
    );
}
