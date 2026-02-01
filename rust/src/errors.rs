use std::fmt;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum EvaluationError {
    /// Users' objective returned an error
    User(Arc<dyn std::error::Error + Send + Sync + 'static>),

    /// Non finite gradient
    NonFiniteGradient,

    /// Non finite value
    NonFiniteValue,
}

impl EvaluationError {
    /// Create an evaluation error from any error type
    pub fn user<E>(error: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::User(Arc::new(error))
    }

    /// Create an evaluation error from a string message
    ///
    /// Use this error if the error type doesn't implement `std::error::Error`
    /// Note: if possible, use `EvaluationError::user` to preserve the error chain.
    pub fn message(msg: impl Into<String>) -> Self {
        Self::User(Arc::new(StringError(msg.into())))
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::User(e) => write!(f, "Evaluation failed:: {e}"),
            Self::NonFiniteValue => write!(f, "Evaluation failed::NonFiniteValue"),
            Self::NonFiniteGradient => write!(f, "Evaluation failed::NonFiniteGradient"),
        }
    }
}

impl std::error::Error for EvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::User(e) => Some(e.as_ref()),
            Self::NonFiniteValue | Self::NonFiniteGradient => None,
        }
    }
}

impl From<std::convert::Infallible> for EvaluationError {
    fn from(err: std::convert::Infallible) -> Self {
        match err {}
    }
}

/// A simple string error for the case where only a message is available
#[derive(Debug, Clone)]
struct StringError(String);

impl fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StringError {}

/// Errors that can occur when calling `tell()`
#[derive(Clone, Debug)]
pub enum TellError {
    /// Called `tell()` when the algorithm has already terminated
    AlreadyTerminated,
    /// Number of results doesn't match number of requested points
    ResultCountMismatch { expected: usize, got: usize },
    /// Gradient dimension doesn't match point dimension
    GradientDimensionMismatch { expected: usize, got: usize },
    /// Wrapper for `EvaluationError`
    EvaluationFailed(EvaluationError),
}

impl std::error::Error for TellError {}

impl fmt::Display for TellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TellError::AlreadyTerminated => write!(f, "Algorithm already terminated"),
            TellError::ResultCountMismatch { expected, got } => {
                write!(f, "Expected {expected} results, got {got}")
            }
            TellError::GradientDimensionMismatch { expected, got } => {
                write!(f, "Expected gradient dimension {expected}, got {got}")
            }
            TellError::EvaluationFailed(e) => write!(f, "Evaluation failed {e:?}"),
        }
    }
}
