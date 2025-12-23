use crate::optimisers::errors::EvaluationError;
use std::error::Error as StdError;

pub type Point = Vec<f64>;
pub type Gradient = Vec<f64>;

/// Trait for type conversion to GradientEvaluation
pub trait IntoEvaluation<T> {
    fn into_evaluation(self) -> Result<T, EvaluationError>;
}

/// Scalar evaluation result
#[derive(Clone, Debug)]
pub struct ScalarEvaluation(pub f64);

/// Convenience for value
impl ScalarEvaluation {
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl IntoEvaluation<ScalarEvaluation> for f64 {
    fn into_evaluation(self) -> Result<ScalarEvaluation, EvaluationError> {
        Ok(ScalarEvaluation(self))
    }
}

/// Identity conversion for ScalarEvaluation
impl IntoEvaluation<ScalarEvaluation> for ScalarEvaluation {
    fn into_evaluation(self) -> Result<ScalarEvaluation, EvaluationError> {
        Ok(self)
    }
}

/// Evaluation result containing both value and gradient
#[derive(Clone, Debug)]
pub struct GradientEvaluation {
    pub value: f64,
    pub gradient: Gradient,
}

impl GradientEvaluation {
    pub fn new(value: f64, gradient: Vec<f64>) -> GradientEvaluation {
        Self { value, gradient }
    }
}

/// Implement for bare tuples (infallible case)
impl IntoEvaluation<GradientEvaluation> for (f64, Vec<f64>) {
    fn into_evaluation(self) -> Result<GradientEvaluation, EvaluationError> {
        Ok(GradientEvaluation::new(self.0, self.1))
    }
}

/// Identity conversion for GradientEvaluation
impl IntoEvaluation<GradientEvaluation> for GradientEvaluation {
    fn into_evaluation(self) -> Result<GradientEvaluation, EvaluationError> {
        Ok(self)
    }
}

/// Result<f64, E>
impl<E: StdError + Send + Sync + 'static> IntoEvaluation<ScalarEvaluation> for Result<f64, E> {
    fn into_evaluation(self) -> Result<ScalarEvaluation, EvaluationError> {
        self.map(ScalarEvaluation).map_err(EvaluationError::user)
    }
}

/// Result<ScalarEvaluation, E>
impl<E: StdError + Send + Sync + 'static> IntoEvaluation<ScalarEvaluation>
    for Result<ScalarEvaluation, E>
{
    fn into_evaluation(self) -> Result<ScalarEvaluation, EvaluationError> {
        self.map_err(EvaluationError::user)
    }
}

/// Result<(f64, Vec<f64), E>
impl<E: StdError + Send + Sync + 'static> IntoEvaluation<GradientEvaluation>
    for Result<(f64, Vec<f64>), E>
{
    fn into_evaluation(self) -> Result<GradientEvaluation, EvaluationError> {
        self.map(|t| GradientEvaluation::new(t.0, t.1))
            .map_err(EvaluationError::user)
    }
}

/// Result<GradientEvaluation, E>
impl<E: StdError + Send + Sync + 'static> IntoEvaluation<GradientEvaluation>
    for Result<GradientEvaluation, E>
{
    fn into_evaluation(self) -> Result<GradientEvaluation, EvaluationError> {
        self.map_err(EvaluationError::user)
    }
}

// Type Aliases
pub trait ScalarInput: IntoEvaluation<ScalarEvaluation> {}
pub trait GradientInput: IntoEvaluation<GradientEvaluation> {}

impl<T: IntoEvaluation<ScalarEvaluation>> ScalarInput for T {}
impl<T: IntoEvaluation<GradientEvaluation>> GradientInput for T {}
