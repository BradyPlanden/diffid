use diffsol::error::DiffsolError;
use std::cell::RefCell;
use std::ops::RangeInclusive;
use std::sync::Arc;

mod diffsol_problem;

use crate::common::{Bounds, Unbounded};
use crate::cost::CostMetric;
use crate::prelude::{OptimisationResults, Optimiser, Sampler, SamplingResults};
pub use diffsol_problem::DiffsolObjective;

pub struct NoGradient;
pub struct NoFunction;

/// A thread-safe, shared function that computes residuals
pub type VectorFn =
    Arc<dyn Fn(&[f64]) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> + Send + Sync>;

#[derive(Debug)]
pub enum ProblemError {
    DimensionMismatch { expected: usize, got: usize },
    External(Box<dyn std::error::Error + Send + Sync>),
    EvaluationFailed(String),
    SolverError(String),
    BuildFailed(String),
}

impl From<DiffsolError> for ProblemError {
    fn from(e: DiffsolError) -> Self {
        ProblemError::BuildFailed(format!("{e}"))
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for ProblemError {
    fn from(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        ProblemError::External(err)
    }
}

impl std::fmt::Display for ProblemError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EvaluationFailed(msg) => write!(f, "evaluation failed: {msg}"),
            Self::SolverError(msg) => write!(f, "solver failed: {msg}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} elements, got {got}")
            }
            Self::BuildFailed(msg) => write!(f, "build failed: {msg}"),
            Self::External(err) => write!(f, "external error: {err}"),
        }
    }
}

impl std::error::Error for ProblemError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub initial_value: f64,
    pub range: ParameterRange,
}

impl ParameterSpec {
    pub fn new<N: Into<String>, P: Into<ParameterRange>>(
        name: N,
        initial_value: f64,
        range: P,
    ) -> Self {
        Self {
            name: name.into(),
            initial_value,
            range: range.into(),
        }
    }

    pub fn bounded<N: Into<String>>(name: N, initial_value: f64, bounds: (f64, f64)) -> Self {
        Self {
            name: name.into(),
            initial_value,
            range: (bounds.0..=bounds.1).into(),
        }
    }

    pub fn unbounded<N: Into<String>>(name: N, initial_value: f64) -> Self {
        Self {
            name: name.into(),
            initial_value,
            range: (f64::NEG_INFINITY..=f64::INFINITY).into(),
        }
    }
}

#[derive(Clone, Default)]
pub struct ParameterSet(Vec<ParameterSpec>);

impl ParameterSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn specs(&self) -> &[ParameterSpec] {
        &self.0
    }

    pub fn push(&mut self, spec: ParameterSpec) {
        self.0.push(spec);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    pub fn take(&mut self) -> Vec<ParameterSpec> {
        std::mem::take(&mut self.0)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ParameterSpec> {
        self.0.iter()
    }

    pub fn bounds(&self) -> Bounds {
        if self.0.is_empty() {
            return Bounds::unbounded(0);
        }

        Bounds {
            limits: self
                .0
                .iter()
                .map(|spec| spec.range.as_inner().clone())
                .collect(),
        }
    }

    pub fn initial_values(&self) -> Vec<f64> {
        if self.0.is_empty() {
            return vec![];
        }
        self.0.iter().map(|spec| spec.initial_value).collect()
    }
}

impl<'a> IntoIterator for &'a ParameterSet {
    type Item = &'a ParameterSpec;
    type IntoIter = std::slice::Iter<'a, ParameterSpec>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterRange(RangeInclusive<f64>);

impl ParameterRange {
    /// Get a reference to the inner range
    pub fn as_inner(&self) -> &RangeInclusive<f64> {
        &self.0
    }

    /// Convert into the inner range
    pub fn into_inner(self) -> RangeInclusive<f64> {
        self.0
    }
}

impl AsRef<RangeInclusive<f64>> for ParameterRange {
    fn as_ref(&self) -> &RangeInclusive<f64> {
        &self.0
    }
}

impl std::ops::Deref for ParameterRange {
    type Target = RangeInclusive<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Implement for tuple
impl From<(f64, f64)> for ParameterRange {
    fn from(bounds: (f64, f64)) -> Self {
        Self(bounds.0..=bounds.1)
    }
}

/// Implement for Unbounded
impl From<Unbounded> for ParameterRange {
    fn from(_: Unbounded) -> Self {
        Self(f64::NEG_INFINITY..=f64::INFINITY)
    }
}

/// Implement for `RangeInclusive`
impl From<RangeInclusive<f64>> for ParameterRange {
    fn from(range: RangeInclusive<f64>) -> Self {
        Self(range)
    }
}

/// An Objective trait, used to define the core
/// evaluation of a `problem`.
pub trait Objective: Send + Sync {
    /// Evaluate the objective function at point `x`
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError`] if evaluation fails:
    /// - [`ProblemError::EvaluationFailed`]: Objective function computation fails
    /// - [`ProblemError::SolverError`]: ODE/DAE solver fails to converge
    /// - [`ProblemError::DimensionMismatch`]: Input or output dimensions are incorrect
    /// - [`ProblemError::External`]: External function propagates an error
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError>;

    fn gradient(&self, _x: &[f64]) -> Option<Vec<f64>> {
        None
    }

    /// Returns true is the objective provides gradients
    fn has_gradient(&self) -> bool {
        false
    }

    /// Evaluate the objective function and its gradient at point `x`
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError`] if evaluation fails:
    /// - [`ProblemError::EvaluationFailed`]: Objective or gradient computation fails
    /// - [`ProblemError::SolverError`]: ODE/DAE solver fails to converge
    /// - [`ProblemError::DimensionMismatch`]: Input or output dimensions are incorrect
    /// - [`ProblemError::External`]: External function propagates an error
    fn evaluate_with_gradient(&self, x: &[f64]) -> Result<(f64, Option<Vec<f64>>), ProblemError> {
        Ok((self.evaluate(x)?, self.gradient(x)))
    }

    /// Evaluates a batch of candidates provided as `xs`
    /// Default is sequential evaluation, certain objectives
    /// support parallel evaluation.
    fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
        xs.iter().map(|x| self.evaluate(x)).collect()
    }

    /// Boolean flag to denote whether the objective supports batched evaluation.
    fn supports_batch_evaluation(&self) -> bool {
        false
    }

    /// Boolean flag to denote whether the objective can support
    /// parallel evaluations.
    ///
    /// When true, `Problem::optimise`/`sample` may route through population
    /// evaluation paths for algorithms that benefit from batching.
    fn supports_parallel_evaluation(&self) -> bool {
        false
    }
}

pub struct ScalarObjective<F = NoFunction, G = NoGradient> {
    f: F,
    grad: G,
}

impl<F> ScalarObjective<F, NoGradient>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self {
            f,
            grad: NoGradient,
        }
    }
}
impl<F, G> ScalarObjective<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    pub fn with_gradient(f: F, grad: G) -> Self {
        Self { f, grad }
    }
}

/// Implement Objective for `NoGradient` state
impl<F> Objective for ScalarObjective<F, NoGradient>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        Ok((self.f)(x))
    }

    fn has_gradient(&self) -> bool {
        false
    }

    fn supports_batch_evaluation(&self) -> bool {
        true
    }
}

/// Implement Objective for gradient state
impl<F, G> Objective for ScalarObjective<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        Ok((self.f)(x))
    }

    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        Some((self.grad)(x))
    }

    fn has_gradient(&self) -> bool {
        true
    }

    fn supports_batch_evaluation(&self) -> bool {
        true
    }
}

pub struct VectorObjective {
    f: VectorFn,
    data: Vec<f64>,
    costs: Vec<Arc<dyn CostMetric>>,
}

thread_local! {
    static VECTOR_RESIDUAL_BUFFER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

impl VectorObjective {
    /// Create a new vector objective
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError::BuildFailed`] if the data vector is empty.
    pub fn new(
        f: VectorFn,
        data: Vec<f64>,
        costs: Vec<Arc<dyn CostMetric>>,
    ) -> Result<Self, ProblemError> {
        if data.is_empty() {
            return Err(ProblemError::BuildFailed(
                "Data must contain at least one element".to_string(),
            ));
        }
        Ok(Self { f, data, costs })
    }
}

impl Objective for VectorObjective {
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        let pred = (self.f)(x).map_err(ProblemError::from)?;

        if pred.len() != self.data.len() {
            return Err(ProblemError::DimensionMismatch {
                expected: self.data.len(),
                got: pred.len(),
            });
        }

        let total_cost = VECTOR_RESIDUAL_BUFFER.with(|buffer| {
            let mut residuals = buffer.borrow_mut();
            residuals.clear();
            let capacity = residuals.capacity();
            if capacity < pred.len() {
                residuals.reserve(pred.len() - capacity);
            }
            residuals.extend(
                pred.iter()
                    .zip(self.data.iter())
                    .map(|(pred, data)| pred - data),
            );
            self.costs.iter().map(|c| c.evaluate(&residuals)).sum()
        });

        Ok(total_cost)
    }

    fn supports_batch_evaluation(&self) -> bool {
        true
    }
}

pub struct Problem<O: Objective> {
    objective: O,
    parameters: ParameterSet,
    optimiser: Optimiser,
    sampler: Sampler,
}

impl<O: Objective> Problem<O> {
    pub fn new(objective: O, parameters: ParameterSet) -> Self {
        Self {
            objective,
            parameters,
            optimiser: Optimiser::default(),
            sampler: Sampler::default(),
        }
    }

    #[must_use]
    pub fn with_optimiser(mut self, optimiser: impl Into<Optimiser>) -> Self {
        self.optimiser = optimiser.into();
        self
    }

    pub fn has_gradient(&self) -> bool {
        self.objective.has_gradient()
    }

    /// Evaluate the problem at point `x`
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError`] if evaluation fails. See [`Objective::evaluate`] for details.
    pub fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        self.objective.evaluate(x)
    }

    /// Evaluate the problem and its gradient at point `x`
    ///
    /// # Errors
    ///
    /// Returns [`ProblemError`] if evaluation fails. See [`Objective::evaluate_with_gradient`] for details.
    pub fn evaluate_with_gradient(
        &self,
        x: &[f64],
    ) -> Result<(f64, Option<Vec<f64>>), ProblemError> {
        self.objective.evaluate_with_gradient(x)
    }

    pub fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
        self.objective.evaluate_population(xs)
    }

    pub fn dimensions(&self) -> usize {
        self.parameters.len()
    }

    pub fn default_parameters(&self) -> Vec<f64> {
        self.parameters.iter().map(|s| s.initial_value).collect()
    }

    pub fn bounds(&self) -> Bounds {
        self.parameters.bounds()
    }

    pub fn initial_values(&self) -> Vec<f64> {
        self.parameters.initial_values()
    }

    fn supports_population_routing(&self) -> bool {
        self.objective.supports_batch_evaluation() || self.objective.supports_parallel_evaluation()
    }

    /// A convenience function for optimisation of the problem
    pub fn optimise(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<&Optimiser>,
    ) -> OptimisationResults {
        let x0 = initial.unwrap_or_else(|| self.default_parameters());

        // Use provided optimiser or,
        // fall back to self.optimiser or,
        // the default
        let opt = optimiser
            .or(Some(&self.optimiser))
            .cloned()
            .unwrap_or_default();

        match opt {
            Optimiser::Scalar(scalar_opt) => {
                if self.supports_population_routing() && scalar_opt.benefits_from_batching() {
                    scalar_opt.run_batch(
                        |xs| self.evaluate_population(xs),
                        x0,
                        self.parameters.bounds(),
                    )
                } else {
                    scalar_opt.run(|x| self.evaluate(x), x0, self.parameters.bounds())
                }
            }
            Optimiser::Gradient(grad_opt) => {
                // Needs objective + gradient values
                // First, check if we have gradients
                if self.has_gradient() {
                    grad_opt.run(
                        |x| {
                            let (value, grad) = self.evaluate_with_gradient(x)?;
                            grad.map_or_else(
                                || {
                                    Err(ProblemError::EvaluationFailed(
                                        "Gradient optimiser requires gradient".to_string(),
                                    ))
                                },
                                |grad| Ok((value, grad)),
                            )
                        },
                        x0,
                        self.parameters.bounds(),
                    )
                } else {
                    // Fall back to numerical gradients
                    grad_opt.run_with_numerical_gradient(
                        |x| self.evaluate(x),
                        x0,
                        self.parameters.bounds(),
                        1e-8,
                    )
                }
            }
        }
    }

    /// A convenience function for sampling the posterior
    pub fn sample(&self, initial: Option<Vec<f64>>, sampler: Option<&Sampler>) -> SamplingResults {
        let x0 = initial.unwrap_or_else(|| self.default_parameters());

        let sampler = sampler.or(Some(&self.sampler)).cloned().unwrap_or_default();

        match sampler {
            Sampler::Scalar(scalar_sampler) => {
                if self.supports_population_routing() && scalar_sampler.benefits_from_batching() {
                    scalar_sampler.run_batch(
                        |xs| self.evaluate_population(xs),
                        x0,
                        self.parameters.bounds(),
                    )
                } else {
                    scalar_sampler.run(|x| self.evaluate(x), x0, self.parameters.bounds())
                }
            }
            Sampler::Gradient(_grad_sampler) => {
                unimplemented!("Gradient samplers not yet implemented")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{RootMeanSquaredError, SumSquaredError};
    use crate::optimisers::CMAES;
    use crate::sampler::MetropolisHastings;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn vector_problem_exponential_model() {
        // Exponential growth: y = y0 * exp(r * t)
        let t_span: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let true_r = 1.5;
        let true_y0 = 2.0;
        let data: Vec<f64> = t_span
            .iter()
            .map(|&t| true_y0 * (true_r * t).exp())
            .collect();

        let t_span_clone = t_span.clone();
        let objective_fn: VectorFn = Arc::new(
            move |params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                let r = params[0];
                let y0 = params[1];
                Ok(t_span_clone.iter().map(|&t| y0 * (r * t).exp()).collect())
            },
        );

        let objective = VectorObjective::new(
            objective_fn,
            data,
            vec![Arc::new(SumSquaredError::default())],
        )
        .expect("failed to create vector objective");

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("r", 1.0, 0.0..=3.0));
        params.push(ParameterSpec::new("y0", 1.0, 0.0..=5.0));

        let problem = Problem::new(objective, params);

        // Test with true parameters
        let cost = problem
            .evaluate(&[true_r, true_y0])
            .expect("evaluation failed");
        assert!(
            cost.abs() < 1e-10,
            "expected near-zero cost with true params, got {cost}"
        );

        // Test with wrong parameters
        let cost = problem.evaluate(&[1.0, 1.0]).expect("evaluation failed");
        assert!(cost > 0.0, "expected positive cost with wrong params");
    }

    #[test]
    fn vector_problem_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let objective_fn: VectorFn = Arc::new(
            |_params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                Ok(vec![1.0, 2.0, 3.0, 4.0, 5.0]) // Wrong size!
            },
        );

        let objective = VectorObjective::new(
            objective_fn,
            data,
            vec![Arc::new(SumSquaredError::default())],
        )
        .expect("failed to create vector objective");

        let problem = Problem::new(objective, ParameterSet::new());

        let result = problem.evaluate(&[1.0]);
        assert!(result.is_err(), "expected error for dimension mismatch");
        match result {
            Err(ProblemError::DimensionMismatch { expected, got }) => {
                assert_eq!(expected, 3);
                assert_eq!(got, 5);
            }
            _ => panic!("expected DimensionMismatch error"),
        }
    }

    #[test]
    fn vector_problem_population_evaluation() {
        let data = vec![1.0, 2.0, 3.0];
        let objective_fn: VectorFn = Arc::new(
            |params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                let scale = params[0];
                Ok(vec![scale, 2.0 * scale, 3.0 * scale])
            },
        );

        let objective = VectorObjective::new(
            objective_fn,
            data,
            vec![Arc::new(SumSquaredError::default())],
        )
        .expect("failed to create vector objective");

        let problem = Problem::new(objective, ParameterSet::new());

        let population = vec![vec![1.0], vec![0.5], vec![1.5], vec![2.0]];

        let sequential: Vec<f64> = population
            .iter()
            .map(|x| problem.evaluate(x).expect("sequential evaluation failed"))
            .collect();

        let batched: Vec<Result<f64, ProblemError>> = problem.evaluate_population(&population);
        let batched: Vec<f64> = batched
            .into_iter()
            .map(|res| res.expect("batched evaluation failed"))
            .collect();

        assert_eq!(sequential.len(), batched.len());
        for (expected, actual) in sequential.iter().zip(batched.iter()) {
            assert_eq!(expected, actual, "population evaluation mismatch");
        }
    }

    #[test]
    fn vector_problem_with_rmse_cost() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let objective_fn: VectorFn = Arc::new(
            |params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                let offset = params[0];
                Ok(vec![1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset])
            },
        );

        let objective = VectorObjective::new(
            objective_fn,
            data,
            vec![Arc::new(RootMeanSquaredError::default())],
        )
        .expect("failed to create vector objective");

        let problem = Problem::new(objective, ParameterSet::new());

        // Perfect fit
        let cost = problem.evaluate(&[0.0]).expect("evaluation failed");
        assert!(cost.abs() < 1e-10);

        // Offset of 1.0 should give RMSE of 1.0
        let cost = problem.evaluate(&[1.0]).expect("evaluation failed");
        assert!(
            (cost - 1.0).abs() < 1e-10,
            "expected RMSE of 1.0, got {cost}"
        );
    }

    #[test]
    fn vector_problem_empty_data_error() {
        let data = vec![];
        let objective_fn: VectorFn = Arc::new(
            |_params: &[f64]| -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
                Ok(vec![])
            },
        );

        let result = VectorObjective::new(
            objective_fn,
            data,
            vec![Arc::new(SumSquaredError::default())],
        );

        assert!(result.is_err(), "expected error for empty data");
        match result {
            Err(ProblemError::BuildFailed(msg)) => {
                assert!(msg.contains("at least one element"));
            }
            _ => panic!("expected BuildFailed error for empty data"),
        }
    }

    struct BatchRoutingObjective {
        single_calls: Arc<AtomicUsize>,
        batch_calls: Arc<AtomicUsize>,
    }

    impl Objective for BatchRoutingObjective {
        fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            Ok(x.iter().map(|xi| xi * xi).sum())
        }

        fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            xs.iter()
                .map(|x| Ok(x.iter().map(|xi| xi * xi).sum()))
                .collect()
        }

        fn supports_batch_evaluation(&self) -> bool {
            true
        }

        fn supports_parallel_evaluation(&self) -> bool {
            false
        }
    }

    struct ParallelRoutingObjective {
        single_calls: Arc<AtomicUsize>,
        batch_calls: Arc<AtomicUsize>,
    }

    impl Objective for ParallelRoutingObjective {
        fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
            self.single_calls.fetch_add(1, Ordering::Relaxed);
            Ok(x.iter().map(|xi| xi * xi).sum())
        }

        fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
            self.batch_calls.fetch_add(1, Ordering::Relaxed);
            xs.iter()
                .map(|x| Ok(x.iter().map(|xi| xi * xi).sum()))
                .collect()
        }

        fn supports_batch_evaluation(&self) -> bool {
            false
        }

        fn supports_parallel_evaluation(&self) -> bool {
            true
        }
    }

    #[test]
    fn optimise_uses_batch_path_without_parallel_evaluation() {
        let single_calls = Arc::new(AtomicUsize::new(0));
        let batch_calls = Arc::new(AtomicUsize::new(0));

        let objective = BatchRoutingObjective {
            single_calls: Arc::clone(&single_calls),
            batch_calls: Arc::clone(&batch_calls),
        };

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("x", 1.0, Unbounded));

        let problem = Problem::new(objective, params)
            .with_optimiser(CMAES::new().with_max_iter(2).with_seed(7));

        let _ = problem.optimise(Some(vec![1.0]), None);

        assert!(
            batch_calls.load(Ordering::Relaxed) > 0,
            "expected batch objective path to be used"
        );
        assert_eq!(
            single_calls.load(Ordering::Relaxed),
            0,
            "single-point objective path should not be used"
        );
    }

    #[test]
    fn sample_uses_batch_path_without_parallel_evaluation() {
        let single_calls = Arc::new(AtomicUsize::new(0));
        let batch_calls = Arc::new(AtomicUsize::new(0));

        let objective = BatchRoutingObjective {
            single_calls: Arc::clone(&single_calls),
            batch_calls: Arc::clone(&batch_calls),
        };

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("x", 1.0, Unbounded));

        let problem = Problem::new(objective, params);
        let sampler: Sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(1)
            .into();

        let _ = problem.sample(Some(vec![1.0]), Some(&sampler));

        assert!(
            batch_calls.load(Ordering::Relaxed) > 0,
            "expected batch objective path to be used"
        );
        assert_eq!(
            single_calls.load(Ordering::Relaxed),
            0,
            "single-point objective path should not be used"
        );
    }

    #[test]
    fn optimise_uses_population_path_with_parallel_capability() {
        let single_calls = Arc::new(AtomicUsize::new(0));
        let batch_calls = Arc::new(AtomicUsize::new(0));

        let objective = ParallelRoutingObjective {
            single_calls: Arc::clone(&single_calls),
            batch_calls: Arc::clone(&batch_calls),
        };

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("x", 1.0, Unbounded));

        let problem = Problem::new(objective, params)
            .with_optimiser(CMAES::new().with_max_iter(2).with_seed(11));

        let _ = problem.optimise(Some(vec![1.0]), None);

        assert!(
            batch_calls.load(Ordering::Relaxed) > 0,
            "expected population objective path to be used"
        );
        assert_eq!(
            single_calls.load(Ordering::Relaxed),
            0,
            "single-point objective path should not be used"
        );
    }

    #[test]
    fn sample_uses_population_path_with_parallel_capability() {
        let single_calls = Arc::new(AtomicUsize::new(0));
        let batch_calls = Arc::new(AtomicUsize::new(0));

        let objective = ParallelRoutingObjective {
            single_calls: Arc::clone(&single_calls),
            batch_calls: Arc::clone(&batch_calls),
        };

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("x", 1.0, Unbounded));

        let problem = Problem::new(objective, params);
        let sampler: Sampler = MetropolisHastings::new()
            .with_num_chains(2)
            .with_iterations(1)
            .into();

        let _ = problem.sample(Some(vec![1.0]), Some(&sampler));

        assert!(
            batch_calls.load(Ordering::Relaxed) > 0,
            "expected population objective path to be used"
        );
        assert_eq!(
            single_calls.load(Ordering::Relaxed),
            0,
            "single-point objective path should not be used"
        );
    }
}
