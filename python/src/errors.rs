use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use chronopt_core::errors::{EvaluationError as CoreEvaluationError, TellError as CoreTellError};

/// Get the custom exception class from chronopt.errors module
fn get_exception_class<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    let errors_module = PyModule::import(py, "chronopt.errors")?;
    errors_module.getattr(name)
}

/// Convert Rust EvaluationError to Python EvaluationError
pub fn evaluation_error_to_py(err: CoreEvaluationError) -> PyErr {
    Python::attach(|py| {
        match get_exception_class(py, "EvaluationError") {
            Ok(exc_class) => {
                let message = format!("Evaluation failed: {}", err);
                // Create exception with message
                match exc_class.call1((message,)) {
                    Ok(exc_instance) => PyErr::from_value(exc_instance.into()),
                    Err(_) => PyValueError::new_err(format!("Evaluation failed: {}", err)),
                }
            }
            Err(_) => {
                // Fallback to ValueError if custom exception not available
                PyValueError::new_err(format!("Evaluation failed: {}", err))
            }
        }
    })
}

/// Convert build errors to Python BuildError
pub fn build_error_to_py(err: impl std::fmt::Display) -> PyErr {
    Python::attach(|py| match get_exception_class(py, "BuildError") {
        Ok(exc_class) => {
            let message = format!("{}", err);
            match exc_class.call1((message,)) {
                Ok(exc_instance) => PyErr::from_value(exc_instance.into()),
                Err(_) => PyValueError::new_err(format!("{}", err)),
            }
        }
        Err(_) => PyValueError::new_err(format!("{}", err)),
    })
}

/// Convert Rust TellError to Python TellError hierarchy
pub fn tell_error_to_py(err: CoreTellError) -> PyErr {
    Python::attach(|py| {
        match err {
            CoreTellError::ResultCountMismatch { expected, got } => {
                match get_exception_class(py, "ResultCountMismatch") {
                    Ok(exc_class) => {
                        // Call with expected and received arguments
                        match exc_class.call1((expected, got)) {
                            Ok(exc_instance) => PyErr::from_value(exc_instance.into()),
                            Err(_) => PyValueError::new_err(format!(
                                "Expected {} evaluation results, but received {}",
                                expected, got
                            )),
                        }
                    }
                    Err(_) => PyValueError::new_err(format!(
                        "Expected {} evaluation results, but received {}",
                        expected, got
                    )),
                }
            }
            CoreTellError::AlreadyTerminated => {
                match get_exception_class(py, "AlreadyTerminated") {
                    Ok(exc_class) => match exc_class.call0() {
                        Ok(exc_instance) => PyErr::from_value(exc_instance.into()),
                        Err(_) => PyValueError::new_err(
                            "Cannot provide results to an already terminated optimization",
                        ),
                    },
                    Err(_) => PyValueError::new_err(
                        "Cannot provide results to an already terminated optimization",
                    ),
                }
            }
            _ => {
                // For other TellError variants, use base TellError class
                match get_exception_class(py, "TellError") {
                    Ok(exc_class) => {
                        let message = format!("Tell error: {}", err);
                        match exc_class.call1((message,)) {
                            Ok(exc_instance) => PyErr::from_value(exc_instance.into()),
                            Err(_) => PyValueError::new_err(format!("Tell error: {}", err)),
                        }
                    }
                    Err(_) => PyValueError::new_err(format!("Tell error: {}", err)),
                }
            }
        }
    })
}
