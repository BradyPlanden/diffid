mod diffsol;
mod scalar;
mod vector;

// mod builders_old; // ToDo: remove

use crate::problem::{ParameterSet, ProblemError};
pub use diffsol::{DiffsolBackend, DiffsolConfig, DiffsolProblemBuilder};
pub use scalar::ScalarProblemBuilder;
pub use vector::VectorProblemBuilder;

#[derive(Debug, Clone)]
pub enum ProblemBuilderError {
    DimensionMismatch { expected: usize, got: usize },
    MissingVectorFn,
    BuildFailed(String),
    MissingData,
    MissingSystem,
}

impl std::fmt::Display for ProblemBuilderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingData => write!(f, "Missing data"),
            Self::BuildFailed(msg) => write!(f, "build failed: {}", msg),
            Self::MissingSystem => write!(f, "Missing system"),
            Self::MissingVectorFn => write!(f, "Missing vector function"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {} elements, got {}", expected, got)
            }
        }
    }
}

impl From<ProblemError> for ProblemBuilderError {
    fn from(err: ProblemError) -> Self {
        match err {
            ProblemError::DimensionMismatch { expected, got } => {
                ProblemBuilderError::DimensionMismatch { expected, got }
            }
            ProblemError::BuildFailed(msg) => ProblemBuilderError::BuildFailed(msg),
            // For other variants, wrap them in BuildFailed
            ProblemError::External(e) => ProblemBuilderError::BuildFailed(e.to_string()),
            ProblemError::EvaluationFailed(msg) => ProblemBuilderError::BuildFailed(msg),
            ProblemError::SolverError(msg) => ProblemBuilderError::BuildFailed(msg),
        }
    }
}

impl std::error::Error for ProblemBuilderError {}

#[cfg(test)]
mod tests {
    use crate::cost::SumSquaredError;
    use crate::prelude::*;
    use nalgebra::DMatrix;
    use std::collections::HashMap;
    #[test]
    fn test_diffsol_builder() {
        let dsl = r#"
in = [r, k]
r { 1 } k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

        let t_span: Vec<f64> = (0..5).map(|i| i as f64 * 0.1).collect();
        let data_values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let data = DMatrix::from_vec(5, 2, {
            let mut columns = Vec::with_capacity(10);
            columns.extend_from_slice(&t_span);
            columns.extend_from_slice(&data_values);
            columns
        });
        let config = HashMap::from([("rtol".to_string(), 1e-6)]);

        let builder = DiffsolProblemBuilder::new()
            .with_diffsl(dsl.to_string())
            .with_data(data)
            .with_config(config)
            .with_parameter("r", 1.0, None)
            .with_parameter("k", 1.0, None);

        let problem = builder.clone().build().unwrap();
        let problem2 = builder.build().unwrap();

        // Test that we can evaluate the problems
        let x0 = vec![1.0, 1.0]; // r, k parameters
        let cost = problem.evaluate(&x0).unwrap();
        let cost2 = problem.evaluate(&x0).unwrap();

        assert_eq!(cost, cost2);
    }

    #[test]
    fn test_scalar_builder() {
        let problem = ScalarProblemBuilder::new()
            .with_function(|x: &[f64]| x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1])
            .with_gradient(|x: &[f64]| vec![2.0 * x[0] + 3.0 * x[1], 3.0 * x[0] + 4.0 * x[1]])
            .build()
            .expect("build scalar problem");

        let x = [1.0_f64, 2.0_f64];
        let (value, grad) = problem
            .evaluate_with_gradient(&x)
            .expect("evaluation failed");

        let grad = grad.expect("Expected gradient to be available");

        assert!((grad[0] - (2.0 * x[0] + 3.0 * x[1])).abs() < 1e-12);
        assert!((grad[1] - (3.0 * x[0] + 4.0 * x[1])).abs() < 1e-12);
        assert!((value - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_vector_builder() {
        // Simple linear model: y = a * x + b
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let problem = VectorProblemBuilder::new()
            .with_function(|params: &[f64]| {
                let a = params[0];
                let b = params[1];
                Ok((0..5).map(|i| a * (i as f64) + b).collect())
            })
            .with_data(data)
            .with_parameter("a", 1.0, None)
            .with_parameter("b", 1.0, None)
            .with_cost(SumSquaredError::default())
            .build()
            .expect("failed to create vector problem");

        // Perfect fit should have zero cost (a=1, b=1 gives [1,2,3,4,5])
        let cost = problem.evaluate(&[1.0, 1.0]).expect("evaluation failed");
        assert!(cost.abs() < 1e-10, "expected near-zero cost, got {}", cost);

        // Non-perfect fit should have positive cost
        let cost = problem.evaluate(&[0.5, 0.5]).expect("evaluation failed");
        assert!(cost > 0.0, "expected positive cost, got {}", cost);
    }
}
