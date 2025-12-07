use std::collections::HashMap;
use std::sync::Arc;

pub mod builders;
mod diffsol_problem;

pub use crate::cost::{CostMetric, RootMeanSquaredError, SumSquaredError};
pub use builders::{BuilderOptimiserExt, BuilderParameterExt};
pub use builders::{
    DiffsolBackend, DiffsolConfig, DiffsolProblemBuilder, OptimiserSlot, ParameterSet,
    ParameterSpec, ScalarProblemBuilder, VectorProblemBuilder,
};

use diffsol::error::DiffsolError;

#[derive(Debug, Clone)]
pub enum ProblemError {
    DimensionMismatch { expected: usize, got: usize },
    EvaluationFailed(String),
    SolverError(String),
    BuildFailed(String),
}

impl From<DiffsolError> for ProblemError {
    fn from(e: DiffsolError) -> Self {
        ProblemError::BuildFailed(format!("{}", e))
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
        }
    }
}

impl std::error::Error for ProblemError {}

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

pub struct ScalarObjective<F, G = fn(&[f64]) -> Vec<f64>> {
    f: F,
    grad: Option<G>,
}

impl<F> ScalarObjective<F, fn(&[f64]) -> Vec<f64>>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
{
    pub fn new(f: F) -> Self {
        Self { f, grad: None }
    }
}
impl<F, G> ScalarObjective<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    pub fn with_gradient(f: F, grad: G) -> Self {
        Self {
            f,
            grad: Some(grad),
        }
    }
}

impl<F, G> Objective for ScalarObjective<F, G>
where
    F: Fn(&[f64]) -> f64 + Send + Sync,
    G: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        Ok((self.f)(x))
    }
    fn gradient(&self, x: &[f64]) -> Option<Vec<f64>> {
        self.grad.as_ref().map(|g| g(x))
    }
}

pub struct ResidualObjective<F> {
    f: F,
    data: Vec<f64>,
    costs: Vec<Arc<dyn CostMetric>>,
}

impl<F> ResidualObjective<F>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ProblemError> + Send + Sync,
{
    pub fn new(f: F, data: Vec<f64>, costs: Vec<Arc<dyn CostMetric>>) -> Self {
        Self { f, data, costs }
    }
}

impl<F> Objective for ResidualObjective<F>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, ProblemError> + Send + Sync,
{
    fn evaluate(&self, x: &[f64]) -> Result<f64, ProblemError> {
        let pred = (self.f)(x)?;

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
    config: HashMap<String, f64>,
}

impl<O: Objective> Problem<O> {
    pub fn new(objective: O, parameters: ParameterSet) -> Self {
        Self {
            objective,
            parameters,
            config: HashMap::new(),
        }
    }

    pub fn with_config(mut self, config: HashMap<String, f64>) -> Self {
        self.config = config;
        self
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
}
