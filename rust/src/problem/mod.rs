use diffsol::error::DiffsolError;
use std::sync::Arc;

mod diffsol_problem;

pub use crate::cost::CostMetric;
use crate::optimisers::{Bounds, OptimisationResults, Optimiser};
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
        ProblemError::BuildFailed(format!("{}", e))
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
            Self::EvaluationFailed(msg) => write!(f, "evaluation failed: {}", msg),
            Self::SolverError(msg) => write!(f, "solver failed: {}", msg),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {} elements, got {}", expected, got)
            }
            Self::BuildFailed(msg) => write!(f, "build failed: {}", msg),
            Self::External(err) => write!(f, "external error: {}", err),
        }
    }
}

impl std::error::Error for ProblemError {}

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub initial_value: f64,
    pub bounds: Option<(f64, f64)>,
}

impl ParameterSpec {
    pub fn new<N>(name: N, initial_value: f64, bounds: Option<(f64, f64)>) -> Self
    where
        N: Into<String>,
    {
        Self {
            name: name.into(),
            initial_value,
            bounds,
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
        self.0.push(spec)
    }

    pub fn clear(&mut self) {
        self.0.clear()
    }

    pub fn take(&mut self) -> Vec<ParameterSpec> {
        std::mem::take(&mut self.0)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ParameterSpec> {
        self.0.iter()
    }

    pub fn bounds(&self) -> Option<Bounds> {
        if self.0.is_empty() {
            return None;
        }

        let all_bounds: Vec<(f64, f64)> = self
            .0
            .iter()
            .map(|spec| spec.bounds.unwrap_or((f64::NEG_INFINITY, f64::INFINITY)))
            .collect();

        // Only return Some if at least one parameter has actual bounds
        let has_bounds = self.0.iter().any(|spec| spec.bounds.is_some());
        if has_bounds {
            Some(Bounds::new(all_bounds))
        } else {
            None
        }
    }
}

/// An Objective trait, used to define the core
/// evaluation of a `problem`.
pub trait Objective: Send + Sync {
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError>;

    fn gradient(&self, _x: &[f64]) -> Option<Vec<f64>> {
        None
    }
    fn evaluate_with_gradient(&self, x: &[f64]) -> Result<(f64, Option<Vec<f64>>), ProblemError> {
        Ok((self.evaluate(x)?, self.gradient(x)))
    }

    fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, ProblemError>> {
        xs.iter().map(|x| self.evaluate(x)).collect()
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
}

pub struct VectorObjective {
    f: VectorFn,
    data: Vec<f64>,
    costs: Vec<Arc<dyn CostMetric>>,
}

impl VectorObjective {
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

        let residuals: Vec<f64> = pred
            .iter()
            .zip(self.data.iter())
            .map(|(pred, data)| pred - data)
            .collect();

        Ok(self.costs.iter().map(|c| c.evaluate(&residuals)).sum())
    }
}

pub struct Problem<O: Objective> {
    objective: O,
    parameters: ParameterSet,
    optimiser: Optimiser,
}

impl<O: Objective> Problem<O> {
    pub fn new(objective: O, parameters: ParameterSet, optimiser: Optimiser) -> Self {
        Self {
            objective,
            parameters,
            optimiser,
        }
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        self.objective.evaluate(x)
    }

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

    pub fn optimize(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<&Optimiser>,
    ) -> OptimisationResults {
        let x0 = initial.unwrap_or_else(|| self.default_parameters());

        // Use provided optimiser, fall back to self.optimiser, then default
        let opt = optimiser
            .or(Some(&self.optimiser))
            .cloned()
            .unwrap_or_default();

        opt.run(|x| self.evaluate(x), x0, self.parameters.bounds())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{RootMeanSquaredError, SumSquaredError};

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
        params.push(ParameterSpec::new("r", 1.0, Some((0.0, 3.0))));
        params.push(ParameterSpec::new("y0", 1.0, Some((0.0, 5.0))));

        let problem = Problem::new(objective, params, Optimiser::default());

        // Test with true parameters
        let cost = problem
            .evaluate(&[true_r, true_y0])
            .expect("evaluation failed");
        assert!(
            cost.abs() < 1e-10,
            "expected near-zero cost with true params, got {}",
            cost
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

        let problem = Problem::new(objective, ParameterSet::new(), Optimiser::default());

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

        let problem = Problem::new(objective, ParameterSet::new(), Optimiser::default());

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

        let problem = Problem::new(objective, ParameterSet::new(), Optimiser::default());

        // Perfect fit
        let cost = problem.evaluate(&[0.0]).expect("evaluation failed");
        assert!(cost.abs() < 1e-10);

        // Offset of 1.0 should give RMSE of 1.0
        let cost = problem.evaluate(&[1.0]).expect("evaluation failed");
        assert!(
            (cost - 1.0).abs() < 1e-10,
            "expected RMSE of 1.0, got {}",
            cost
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
}
