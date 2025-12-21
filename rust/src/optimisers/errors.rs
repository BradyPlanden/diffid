use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum EvaluationError {
    /// Users' objective returned an error
    User(Arc<dyn std::error::Error + Send + Sync + 'static>),

    /// Non finite gradient
    NonFiniteGradient,
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

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::User(e) => write!(f, "Evaluation failed:: {}", e),
            Self::NonFiniteGradient => write!(f, "Evaluation failed::NonFiniteGradient"),
        }
    }
}

impl std::error::Error for EvaluationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::User(e) => Some(e.as_ref()),
            Self::NonFiniteGradient => None,
        }
    }
}

/// A simple string error for the case where only a message is available
#[derive(Debug, Clone)]
struct StringError(String);

impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for StringError {}
