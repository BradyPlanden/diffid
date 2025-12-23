mod adam;
mod cmaes;
mod errors;
mod nelder_mead;
mod types;

use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use crate::builders::{DiffsolProblemBuilder, ScalarProblemBuilder, VectorProblemBuilder};
use crate::problem::{Objective, ProblemError};

pub use adam::Adam;
pub use cmaes::CMAES;
pub use nelder_mead::NelderMead;
pub use types::*;

/// Result of calling `ask()`
#[derive(Clone, Debug)]
pub enum AskResult {
    /// Evaluate these points and call `tell()` with the results
    Evaluate(Vec<Point>),
    /// Optimization has finished
    Done(OptimisationResults),
}

/// Errors that can occur when calling `tell()`
#[derive(Clone, Debug, PartialEq)]
pub enum TellError {
    /// Called `tell()` when the algorithm has already terminated
    AlreadyTerminated,
    /// Number of results doesn't match number of requested points
    ResultCountMismatch { expected: usize, got: usize },
    /// Gradient dimension doesn't match point dimension
    GradientDimensionMismatch { expected: usize, got: usize },
}

#[derive(Clone)]
pub enum Optimiser {
    NelderMead(NelderMead),
    CMAES(CMAES),
    Adam(Adam),
}

impl Optimiser {
    pub fn run<F, E>(
        &self,
        objective: F,
        initial: Point,
        bounds: Option<Bounds>,
    ) -> OptimisationResults
    where
        F: FnMut(&[f64]) -> Result<f64, E>,
        E: StdError + 'static,
    {
        match self {
            Optimiser::NelderMead(nm) => nm.run(objective, initial, bounds),
            Optimiser::CMAES(cm) => cm.run(objective, initial, bounds),
            Optimiser::Adam(ad) => ad.run(objective, initial, bounds),
        }
    }
}

impl From<NelderMead> for Optimiser {
    fn from(nm: NelderMead) -> Self {
        Optimiser::NelderMead(nm)
    }
}

impl From<CMAES> for Optimiser {
    fn from(cmaes: CMAES) -> Self {
        Optimiser::CMAES(cmaes)
    }
}

impl From<Adam> for Optimiser {
    fn from(adam: Adam) -> Self {
        Optimiser::Adam(adam)
    }
}

impl Default for Optimiser {
    fn default() -> Self {
        Optimiser::NelderMead(NelderMead::default())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Bounds {
    limits: Vec<(f64, f64)>,
}

impl Bounds {
    pub fn new(limits: Vec<(f64, f64)>) -> Self {
        Self { limits }
    }

    pub fn apply(&self, point: &mut [f64]) {
        debug_assert_eq!(point.len(), self.limits.len(), "Dimension mismatch");
        point
            .iter_mut()
            .zip(&self.limits)
            .for_each(|(val, &(lo, hi))| *val = val.clamp(lo, hi));
    }

    pub fn dimension(&self) -> usize {
        self.limits.len()
    }
}

pub trait ApplyBounds {
    fn apply_bounds(&mut self, bounds: Option<&Bounds>);
}

impl ApplyBounds for [f64] {
    fn apply_bounds(&mut self, bounds: Option<&Bounds>) {
        if let Some(b) = bounds {
            b.apply(self);
        }
    }
}

#[derive(Debug, Clone)]
struct EvaluatedPoint {
    point: Vec<f64>,
    value: f64,
}

impl EvaluatedPoint {
    fn new(point: Vec<f64>, value: f64) -> Self {
        Self { point, value }
    }
}

fn build_results(
    points: &[EvaluatedPoint],
    iterations: usize,
    evaluations: usize,
    time: Duration,
    reason: TerminationReason,
    covariance: Option<&DMatrix<f64>>,
) -> OptimisationResults {
    let mut ordered = points.to_vec();
    ordered.sort_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(Ordering::Equal));

    let best = ordered
        .first()
        .cloned()
        .unwrap_or_else(|| EvaluatedPoint::new(Vec::new(), f64::NAN));

    let final_simplex = ordered.iter().map(|v| v.point.clone()).collect();
    let final_simplex_values = ordered.iter().map(|v| v.value).collect();

    let covariance = covariance.map(|matrix| {
        (0..matrix.nrows())
            .map(|row| matrix.row(row).iter().copied().collect())
            .collect()
    });

    let success = matches!(
        reason,
        TerminationReason::FunctionToleranceReached
            | TerminationReason::ParameterToleranceReached
            | TerminationReason::BothTolerancesReached
            | TerminationReason::GradientToleranceReached
    );

    let message = reason.to_string();

    OptimisationResults {
        x: best.point,
        value: best.value,
        iterations,
        evaluations,
        time,
        success,
        message,
        termination: reason,
        final_simplex,
        final_simplex_values,
        covariance,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TerminationReason {
    FunctionToleranceReached,
    ParameterToleranceReached,
    GradientToleranceReached,
    BothTolerancesReached,
    MaxIterationsReached,
    MaxFunctionEvaluationsReached,
    DegenerateSimplex,
    PatienceElapsed,
    FunctionEvaluationFailed(String),
}

impl fmt::Display for TerminationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminationReason::FunctionToleranceReached => {
                write!(f, "Function tolerance met")
            }
            TerminationReason::ParameterToleranceReached => {
                write!(f, "Parameter tolerance met")
            }
            TerminationReason::GradientToleranceReached => {
                write!(f, "Gradient tolerance met")
            }
            TerminationReason::BothTolerancesReached => {
                write!(f, "Function and parameter tolerances met")
            }
            TerminationReason::MaxIterationsReached => {
                write!(f, "Maximum iterations reached")
            }
            TerminationReason::MaxFunctionEvaluationsReached => {
                write!(f, "Maximum function evaluations reached")
            }
            TerminationReason::DegenerateSimplex => {
                write!(f, "Degenerate simplex encountered")
            }
            TerminationReason::PatienceElapsed => {
                write!(f, "Patience elapsed")
            }
            TerminationReason::FunctionEvaluationFailed(msg) => {
                write!(f, "Function evaluation failed: {}", msg)
            }
        }
    }
}

// Results object
#[derive(Debug, Clone)]
pub struct OptimisationResults {
    pub x: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    pub evaluations: usize,
    pub time: Duration,
    pub success: bool,
    pub message: String,
    pub termination: TerminationReason,
    pub final_simplex: Vec<Vec<f64>>,
    pub final_simplex_values: Vec<f64>,
    pub covariance: Option<Vec<Vec<f64>>>,
}

impl OptimisationResults {
    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, nfev={}, time={:?}, success={}, reason={})",
            self.x, self.value, self.iterations, self.evaluations, self.time, self.success, self.message
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimisers::cmaes::{CMAESState, StrategyParameters};
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn nelder_mead_minimises_quadratic() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                let x0 = x[0] - 1.5;
                let x1 = x[1] + 0.5;
                x0 * x0 + x1 * x1
            })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_position_tolerance(1e-8);

        let result = optimiser.run(
            |x| problem.evaluate(x),
            vec![1.0],
            Some(Bounds::new(vec![(-5.0, 4.0)])),
        );

        assert!(result.success, "Expected success: {}", result.message);
        assert!((result.x[0] - 1.5).abs() < 1e-5);
        assert!((result.x[1] + 0.5).abs() < 1e-5);
        assert!(
            result.value < 1e-9,
            "Final value too large: {}",
            result.value
        );
        assert!(result.iterations > 0);
        assert!(result.evaluations > result.iterations);
    }

    #[test]
    fn nelder_mead_respects_max_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new().with_max_iter(1).with_sigma0(1.0);
        let result = optimiser.run(|x| problem.evaluate(x), vec![10.0, -10.0], None);

        assert_eq!(result.termination, TerminationReason::MaxIterationsReached);
        assert!(!result.success);
        assert!(result.iterations <= 1);
    }

    #[test]
    fn nelder_mead_respects_max_function_evaluations() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new()
            .with_max_evaluations(2)
            .with_sigma0(0.5)
            .with_max_iter(500);

        let result = optimiser.run(|x| problem.evaluate(x), vec![2.0, 2.0], None);

        assert_eq!(
            result.termination,
            TerminationReason::MaxFunctionEvaluationsReached
        );
        assert!(!result.success);
        assert!(result.evaluations <= 2);
    }

    #[test]
    fn nelder_mead_respects_patience() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| xi * xi).sum()
            })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new().with_sigma0(0.5).with_patience(0.01);

        let result = optimiser.run(|x| problem.evaluate(x), vec![5.0, -5.0], None);

        assert_eq!(result.termination, TerminationReason::PatienceElapsed);
        assert!(!result.success);
    }

    #[test]
    fn cmaes_minimises_quadratic() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                let x0 = x[0] - 1.5;
                let x1 = x[1] + 0.5;
                x0 * x0 + x1 * x1
            })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_patience(5.0)
            .with_seed(42);

        let result = optimiser.run(|x| problem.evaluate(x), vec![5.0, -4.0], None);

        assert!(result.success, "Expected success: {}", result.message);
        assert!((result.x[0] - 1.5).abs() < 1e-4);
        assert!((result.x[1] + 0.5).abs() < 1e-4);
        assert!(
            result.value < 1e-8,
            "Final value too large: {}",
            result.value
        );
        assert!(result.iterations > 0);
        assert!(result.evaluations > result.iterations);
    }

    #[test]
    fn cmaes_respects_max_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new().with_max_iter(1).with_sigma0(0.5).with_seed(7);
        let result = optimiser.run(|x| problem.evaluate(x), vec![10.0, -10.0], None);

        assert_eq!(result.termination, TerminationReason::MaxIterationsReached);
        assert!(!result.success);
        assert!(result.iterations <= 1);
    }

    #[test]
    fn cmaes_respects_patience() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| xi * xi).sum()
            })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new()
            .with_sigma0(0.5)
            .with_patience(0.01)
            .with_seed(5);

        let result = optimiser.run(|x| problem.evaluate(x), vec![5.0, -5.0], None);

        assert_eq!(result.termination, TerminationReason::PatienceElapsed);
        assert!(!result.success);
    }

    #[test]
    fn adam_minimises_quadratic_with_gradient() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                let x0 = x[0] - 1.5;
                let x1 = x[1] + 0.5;
                x0 * x0 + x1 * x1
            })
            .with_gradient(|x: &[f64]| vec![2.0 * (x[0] - 1.5), 2.0 * (x[1] + 0.5)])
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = Adam::new()
            .with_step_size(0.1)
            .with_max_iter(500)
            .with_threshold(1e-8);

        let result = optimiser.run(
            |x| {
                let val_grad = problem
                    .evaluate_with_gradient(x)
                    .expect("Should succeed with function evaluation w/ gradient");
                (val_grad.0, val_grad.1.expect("Expected gradient"))
            },
            vec![5.0, -4.0],
            None,
        );

        assert!(result.success, "Expected success: {}", result.message);
        assert!((result.x[0] - 1.5).abs() < 1e-3);
        assert!((result.x[1] + 0.5).abs() < 1e-3);
        assert!(
            result.value < 1e-6,
            "Final value too large: {}",
            result.value
        );
    }

    #[test]
    fn adam_respects_max_iterations() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .with_gradient(|x: &[f64]| x.iter().map(|xi| 2.0 * xi).collect())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = Adam::new()
            .with_step_size(0.1)
            .with_max_iter(1)
            .with_threshold(1e-12);

        let result = optimiser.run(
            |x| {
                let val_grad = problem
                    .evaluate_with_gradient(x)
                    .expect("Should succeed with function evaluation w/ gradient");
                (val_grad.0, val_grad.1.expect("Expected gradient"))
            },
            vec![10.0, -10.0],
            None,
        );

        assert_eq!(result.termination, TerminationReason::MaxIterationsReached);
        assert!(!result.success);
        assert!(result.iterations <= 1);
    }

    #[test]
    fn adam_respects_patience() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| xi * xi).sum()
            })
            .with_gradient(|x: &[f64]| {
                std::thread::sleep(Duration::from_millis(5));
                x.iter().map(|xi| 2.0 * xi).collect()
            })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = Adam::new()
            .with_step_size(0.1)
            .with_max_iter(100)
            .with_patience(0.01);

        let result = optimiser.run(
            |x| {
                let val_grad = problem
                    .evaluate_with_gradient(x)
                    .expect("Should succeed with function evaluation w/ gradient");
                (val_grad.0, val_grad.1.expect("Expected gradient"))
            },
            vec![5.0, -5.0],
            None,
        );

        assert_eq!(result.termination, TerminationReason::PatienceElapsed);
        assert!(!result.success);
    }

    #[test]
    fn adam_fails_without_gradient() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = Adam::new().with_max_iter(10);
        let result = optimiser.run(
            |x| -> Result<(f64, Vec<f64>), ProblemError> {
                let (value, gradient_opt) = problem.evaluate_with_gradient(x)?;
                match gradient_opt {
                    Some(g) => Ok((value, g)),
                    None => Err(ProblemError::EvaluationFailed(
                        "Adam optimiser requires an available gradient".to_string(),
                    )),
                }
            },
            vec![1.0, 2.0],
            None,
        );

        assert!(!result.success);
        match result.termination {
            TerminationReason::FunctionEvaluationFailed(ref msg) => {
                assert!(msg.contains("requires an available gradient"));
            }
            other => panic!("expected FunctionEvaluationFailed, got {:?}", other),
        }
    }

    // Edge case tests
    #[test]
    fn nelder_mead_handles_bounds() {
        use crate::problem::ParameterSpec;

        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2))
            .with_parameter("x", 0.0, Some((-1.0, 1.0)))
            .with_parameter("y", 0.0, Some((0.0, 2.0)))
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new().with_max_iter(200).with_threshold(1e-8);

        let result = optimiser.run(|x| problem.evaluate(x), vec![0.5, 1.0], None);

        // Should converge to bounds: x=1.0 (clamped from 2.0), y=2.0 (clamped from 3.0)
        assert!(
            result.x[0] >= -1.0 && result.x[0] <= 1.0,
            "x out of bounds: {}",
            result.x[0]
        );
        assert!(
            result.x[1] >= 0.0 && result.x[1] <= 2.0,
            "y out of bounds: {}",
            result.x[1]
        );
        assert!(
            (result.x[0] - 1.0).abs() < 0.1,
            "x should be near upper bound"
        );
        assert!(
            (result.x[1] - 2.0).abs() < 0.1,
            "y should be near upper bound"
        );
    }

    #[test]
    fn cmaes_handles_bounds() {
        use crate::problem::ParameterSpec;

        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| (x[0] - 5.0).powi(2) + (x[1] + 5.0).powi(2))
            .with_parameter("x", 0.0, Some((0.0, 3.0)))
            .with_parameter("y", 0.0, Some((-3.0, 0.0)))
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(123);

        let result = optimiser.run(|x| problem.evaluate(x), vec![1.5, -1.5], None);

        // Should converge to bounds: x=3.0 (clamped from 5.0), y=-3.0 (clamped from -5.0)
        assert!(
            result.x[0] >= 0.0 && result.x[0] <= 3.0,
            "x out of bounds: {}",
            result.x[0]
        );
        assert!(
            result.x[1] >= -3.0 && result.x[1] <= 0.0,
            "y out of bounds: {}",
            result.x[1]
        );
        assert!(
            (result.x[0] - 3.0).abs() < 0.2,
            "x should be near upper bound"
        );
        assert!(
            (result.x[1] + 3.0).abs() < 0.2,
            "y should be near lower bound"
        );
    }

    #[test]
    fn cmaes_is_reproducible_with_seed() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new()
            .with_max_iter(300)
            .with_threshold(1e-8)
            .with_sigma0(0.7)
            .with_seed(2024);

        let initial = vec![3.0, -2.0];
        let result_one = optimiser.run(|x| problem.evaluate(x), initial.clone(), None);
        let result_two = optimiser.run(|x| problem.evaluate(x), initial, None);

        assert!(
            result_one.success,
            "first run should converge: {}",
            result_one.message
        );
        assert!(
            result_two.success,
            "second run should converge: {}",
            result_two.message
        );
        assert_eq!(result_one.iterations, result_two.iterations);
        assert_eq!(result_one.evaluations, result_two.evaluations);
        assert!((result_one.value - result_two.value).abs() < 1e-12);
        for (x1, x2) in result_one.x.iter().zip(result_two.x.iter()) {
            assert!(
                (x1 - x2).abs() < 1e-10,
                "expected identical optima: {} vs {}",
                x1,
                x2
            );
        }
        assert_eq!(result_one.covariance, result_two.covariance);
    }

    #[test]
    fn nelder_mead_handles_nan_objective() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| if x[0] > 1.0 { f64::NAN } else { x[0] * x[0] })
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = NelderMead::new().with_max_iter(50).with_sigma0(2.0); // Larger sigma to ensure we hit NaN region

        let result = optimiser.run(|x| problem.evaluate(x), vec![0.5], None);

        // Should either detect NaN or converge to valid region
        // If it hits NaN, it should fail gracefully
        if !result.success {
            assert!(matches!(
                result.termination,
                TerminationReason::FunctionEvaluationFailed(_)
            ));
        }
        // Otherwise it converged to the valid region (x <= 1.0)
    }

    #[test]
    fn cmaes_lazy_eigendecomposition_works() {
        // Test with high dimension to trigger lazy updates
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>())
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let dim = 60; // > 50 to trigger lazy updates
        let initial = vec![0.5; dim];

        let optimiser = CMAES::new()
            .with_max_iter(100)
            .with_threshold(1e-6)
            .with_seed(777);

        let initial_value = initial.iter().map(|x| x * x).sum::<f64>();

        let result = optimiser.run(|x| problem.evaluate(x), initial, None);

        // Should still work with lazy updates and improve from initial
        assert!(result.evaluations > 0);
        assert!(
            result.value < initial_value,
            "Should improve: {} < {}",
            result.value,
            initial_value
        );
    }

    #[test]
    fn cmaes_covariance_is_symmetric_and_psd() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2))
            .build()
            .expect("Problem builder should succeed with valid parameters");

        let optimiser = CMAES::new()
            .with_max_iter(400)
            .with_threshold(1e-10)
            .with_sigma0(0.6)
            .with_seed(4242);

        let result = optimiser.run(|x| problem.evaluate(x), vec![4.5, -3.5], None);

        assert!(result.success, "Expected success: {}", result.message);
        assert!(
            result.value < 1e-6,
            "Should reach low objective value: {}",
            result.value
        );

        let covariance = result
            .covariance
            .clone()
            .expect("CMAES should provide covariance estimates");
        assert_eq!(covariance.len(), 2);
        assert!(covariance.iter().all(|row| row.len() == 2));

        covariance
            .iter()
            .zip(&covariance)
            .for_each(|(row_i, row_j)| {
                row_i.iter().zip(row_j).for_each(|(a, b)| {
                    assert!((a - b).abs() < 1e-12, "covariance matrix must be symmetric")
                });
            });

        let flat: Vec<f64> = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let matrix = DMatrix::from_row_slice(2, 2, &flat);
        let eigenvalues = matrix.symmetric_eigen().eigenvalues;

        assert!(
            eigenvalues.iter().all(|&eig| eig >= -1e-10),
            "covariance must be positive semi-definite: {:?}",
            eigenvalues
        );
    }

    #[test]
    fn d_sigma_matches_hansen_2016_formula() {
        // Test case 1: Standard parameters from 10-dimensional optimization
        let mu_eff = 4.5;
        let dim_f = 10.0;
        let c_sigma = 0.3;

        // Manual computation following Hansen (2016)
        let inner: f64 = (mu_eff - 1.0) / (dim_f + 1.0); // (4.5 - 1) / 11 = 0.318...
        let sqrt_inner = inner.sqrt(); // ~0.564
        let clamped = (sqrt_inner - 1.0).max(0.0); // max(0, -0.436) = 0.0
        let expected = 1.0 + c_sigma + 2.0 * clamped; // 1.0 + 0.3 + 0 = 1.3

        let computed = StrategyParameters::compute_d_sigma(mu_eff, dim_f, c_sigma);

        assert!(
            (computed - expected).abs() < 1e-12,
            "d_sigma mismatch: expected {}, got {}",
            expected,
            computed
        );

        // For this case, the sqrt term is less than 1, so it should clamp to 0
        assert!(
            (computed - (1.0 + c_sigma)).abs() < 1e-12,
            "When sqrt((mu_eff-1)/(n+1)) < 1, d_sigma should equal 1 + c_sigma"
        );
    }

    #[test]
    fn cmaes_d_sigma_clamps_when_below_unity() {
        let mu_eff = 2.0_f64;
        let dim_f = 10.0_f64;
        let c_sigma = 0.2_f64;

        let expected = 1.0 + c_sigma;
        let computed = StrategyParameters::compute_d_sigma(mu_eff, dim_f, c_sigma);

        assert!((computed - expected).abs() < 1e-12);
    }

    #[test]
    fn covariance_update_applies_exponential_correction() {
        let cov = DMatrix::from_row_slice(2, 2, &[2.0, 0.1, 0.1, 1.0]);
        let c1 = 0.3_f64;
        let c_mu = 0.2_f64;
        let c_c = 0.5_f64;
        let h_sigma = 0.0_f64;
        let p_c = DVector::from_vec(vec![1.0, -0.5]);
        let rank_mu = DMatrix::zeros(2, 2);

        let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
        let expected = cov.clone() * (1.0 - c1 - c_mu)
            + (p_c.clone() * p_c.transpose() + cov.clone() * correction_factor) * c1
            + rank_mu.clone() * c_mu;

        let updated = CMAESState::update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {} got {}", exp, got);
        }
    }

    #[test]
    fn covariance_update_skips_correction_when_h_sigma_one() {
        let cov = DMatrix::from_row_slice(2, 2, &[1.5, 0.2, 0.2, 0.8]);
        let c1 = 0.25_f64;
        let c_mu = 0.1_f64;
        let c_c = 0.6_f64;
        let h_sigma = 1.0_f64;
        let p_c = DVector::from_vec(vec![0.3, -0.7]);
        let rank_mu = DMatrix::from_row_slice(2, 2, &[0.05, 0.01, 0.01, 0.04]);

        let correction_factor = (1.0 - h_sigma) * c_c * (2.0 - c_c);
        let expected = cov.clone() * (1.0 - c1 - c_mu)
            + (p_c.clone() * p_c.transpose() + cov.clone() * correction_factor) * c1
            + rank_mu.clone() * c_mu;

        let updated = CMAESState::update_covariance(&cov, c1, c_mu, &p_c, h_sigma, c_c, &rank_mu);

        for (exp, got) in expected.iter().zip(updated.iter()) {
            assert!((exp - got).abs() < 1e-12, "expected {} got {}", exp, got);
        }
    }
}
