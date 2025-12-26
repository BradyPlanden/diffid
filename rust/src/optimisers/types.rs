use crate::optimisers::errors::EvaluationError;
use std::error::Error as StdError;

pub type Gradient = Vec<f64>;

/// Scalar evaluation result
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ScalarEvaluation(pub f64);

/// Convenience for value
impl ScalarEvaluation {
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if finite
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
}

/// TryFrom f64 with validation
impl TryFrom<f64> for ScalarEvaluation {
    type Error = EvaluationError;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        if value.is_finite() {
            Ok(ScalarEvaluation(value))
        } else {
            Err(EvaluationError::NonFiniteValue)
        }
    }
}

/// From Result<f64, E> for error propagation
impl<E: StdError + Send + Sync + 'static> TryFrom<Result<f64, E>> for ScalarEvaluation {
    type Error = EvaluationError;
    fn try_from(result: Result<f64, E>) -> Result<Self, Self::Error> {
        match result {
            Ok(value) => ScalarEvaluation::try_from(value),
            Err(err) => Err(EvaluationError::user(err)),
        }
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

    pub fn is_finite(&self) -> bool {
        self.value.is_finite() && self.gradient.iter().all(|g| g.is_finite())
    }
}

/// TryFrom (f64, Vec<f64>) with validation
impl TryFrom<(f64, Vec<f64>)> for GradientEvaluation {
    type Error = EvaluationError;

    fn try_from(tuple: (f64, Vec<f64>)) -> Result<Self, Self::Error> {
        let eval = GradientEvaluation::new(tuple.0, tuple.1);
        if eval.is_finite() {
            Ok(eval)
        } else {
            Err(EvaluationError::NonFiniteGradient)
        }
    }
}

/// Result<(f64, Vec<f64), E>
impl<E: StdError + Send + Sync + 'static> TryFrom<Result<(f64, Vec<f64>), E>>
    for GradientEvaluation
{
    type Error = EvaluationError;

    fn try_from(result: Result<(f64, Vec<f64>), E>) -> Result<Self, Self::Error> {
        match result {
            Ok(tuple) => GradientEvaluation::try_from(tuple),
            Err(e) => Err(EvaluationError::user(e)),
        }
    }
}