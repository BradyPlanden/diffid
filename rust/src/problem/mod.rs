use crate::optimisers::{NelderMead, OptimisationResults, Optimiser};
use diffsol::OdeBuilder;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Arc;

pub mod builders;
pub mod diffsol_problem;
pub use crate::cost::{CostMetric, RootMeanSquaredError, SumSquaredError};
pub use builders::{BuilderOptimiserExt, BuilderParameterExt};
pub use builders::{
    DiffsolBackend, DiffsolConfig, DiffsolProblemBuilder, OptimiserSlot, ParameterSet,
    ParameterSpec, ScalarProblemBuilder, VectorProblemBuilder,
};
pub use diffsol_problem::DiffsolProblem;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type GradientFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
pub type VectorObjectiveFn = Box<dyn Fn(&[f64]) -> Result<Vec<f64>, String> + Send + Sync>;

pub struct CallableObjective {
    objective: ObjectiveFn,
    gradient: Option<GradientFn>,
}

impl CallableObjective {
    pub(crate) fn new(objective: ObjectiveFn, gradient: Option<GradientFn>) -> Self {
        Self {
            objective,
            gradient,
        }
    }

    fn evaluate(&self, x: &[f64]) -> f64 {
        self.objective.as_ref()(x)
    }

    fn gradient(&self) -> Option<&GradientFn> {
        self.gradient.as_ref()
    }
}

pub struct VectorProblem {
    objective: VectorObjectiveFn,
    data: Vec<f64>,
    shape: Vec<usize>,
    cost_metric: Arc<dyn CostMetric>,
}

impl VectorProblem {
    pub(crate) fn new(
        objective: VectorObjectiveFn,
        data: Vec<f64>,
        shape: Vec<usize>,
        cost_metric: Arc<dyn CostMetric>,
    ) -> Self {
        Self {
            objective,
            data,
            shape,
            cost_metric,
        }
    }

    fn validate_prediction(&self, len: usize) -> Result<(), String> {
        if len == self.data.len() {
            return Ok(());
        }

        let shape_description = if self.shape.is_empty() {
            "unknown".to_string()
        } else {
            format!("{:?}", self.shape)
        };

        Err(format!(
            "Vector objective produced {} elements but data shape {} expects {} elements",
            len,
            shape_description,
            self.data.len()
        ))
    }

    fn evaluate(&self, x: &[f64]) -> Result<f64, String> {
        let prediction = (self.objective)(x)?;
        self.validate_prediction(prediction.len())?;

        let residuals: Vec<f64> = prediction
            .iter()
            .zip(self.data.iter())
            .map(|(pred, obs)| pred - obs)
            .collect();
        Ok(self.cost_metric.evaluate(&residuals))
    }

    fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, String>> {
        xs.iter().map(|params| self.evaluate(params)).collect()
    }
}

pub type SharedOptimiser = Arc<dyn Optimiser + Send + Sync>;

/// Different kinds of problems
pub enum ProblemKind {
    Callable(CallableObjective),
    Diffsol(Box<DiffsolProblem>),
    Vector(VectorProblem),
}
// Problem class
pub struct Problem {
    kind: ProblemKind,
    config: HashMap<String, f64>,
    parameter_specs: ParameterSet,
    default_optimiser: Option<SharedOptimiser>,
}

impl Problem {
    pub fn new_diffsol(
        dsl: &str,
        data: DMatrix<f64>,
        t_span: Vec<f64>,
        config: DiffsolConfig,
        parameter_specs: ParameterSet,
        cost_metric: Arc<dyn CostMetric>,
        default_optimiser: Option<SharedOptimiser>,
    ) -> Result<Self, String> {
        let backend_problem = match config.backend {
            DiffsolBackend::Dense => OdeBuilder::<diffsol::NalgebraMat<f64>>::new()
                .atol([config.atol])
                .rtol(config.rtol)
                .build_from_diffsl(dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| diffsol_problem::BackendProblem::Dense(Box::new(problem))),
            DiffsolBackend::Sparse => OdeBuilder::<diffsol::FaerSparseMat<f64>>::new()
                .atol([config.atol])
                .rtol(config.rtol)
                .build_from_diffsl(dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(|problem| diffsol_problem::BackendProblem::Sparse(Box::new(problem))),
        }?;

        let problem = diffsol_problem::DiffsolProblem::new(
            backend_problem,
            dsl.to_string(),
            config.clone(),
            t_span,
            data,
            cost_metric,
        );

        Ok(Problem {
            kind: ProblemKind::Diffsol(Box::new(problem)),
            config: config.to_map(),
            parameter_specs,
            default_optimiser,
        })
    }

    pub fn new_vector(
        objective: VectorObjectiveFn,
        data: Vec<f64>,
        shape: Vec<usize>,
        config: HashMap<String, f64>,
        parameter_specs: ParameterSet,
        cost_metric: Arc<dyn CostMetric>,
        default_optimiser: Option<SharedOptimiser>,
    ) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Data must contain at least one element".to_string());
        }

        if !shape.is_empty() {
            let expected_len: usize = shape.iter().product();
            if expected_len != data.len() {
                return Err(format!(
                    "Data length {} does not match provided shape {:?} (expected {})",
                    data.len(),
                    shape,
                    expected_len
                ));
            }
        }

        Ok(Problem {
            kind: ProblemKind::Vector(VectorProblem::new(objective, data, shape, cost_metric)),
            config,
            parameter_specs,
            default_optimiser,
        })
    }

    pub fn evaluate(&self, x: &[f64], gradient: bool) -> Result<f64, String> {
        match &self.kind {
            ProblemKind::Callable(callable) => Ok(callable.evaluate(x)),
            ProblemKind::Diffsol(problem) => problem.evaluate(x, gradient),
            ProblemKind::Vector(vector) => vector.evaluate(x),
        }
    }

    pub fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, String>> {
        match &self.kind {
            ProblemKind::Callable(callable) => {
                xs.iter().map(|x| Ok(callable.evaluate(x))).collect()
            }
            ProblemKind::Diffsol(problem) => {
                let slices: Vec<&[f64]> = xs.iter().map(|x| x.as_slice()).collect();
                problem.evaluate_population(&slices)
            }
            ProblemKind::Vector(vector) => vector.evaluate_population(xs),
        }
    }

    pub fn get_config(&self, key: &str) -> Option<&f64> {
        self.config.get(key)
    }

    pub fn config(&self) -> &HashMap<String, f64> {
        &self.config
    }

    pub fn parameter_specs(&self) -> &ParameterSet {
        &self.parameter_specs
    }

    pub fn default_parameters(&self) -> Vec<f64> {
        if self.parameter_specs.is_empty() {
            return Vec::new();
        }

        self.parameter_specs
            .iter()
            .map(|spec| spec.initial_value)
            .collect()
    }

    pub fn dimension(&self) -> usize {
        self.parameter_specs.len()
    }

    pub fn gradient(&self) -> Option<&GradientFn> {
        match &self.kind {
            ProblemKind::Callable(callable) => callable.gradient(),
            ProblemKind::Diffsol(_) => None,
            ProblemKind::Vector(_) => None,
        }
    }

    pub fn optimize(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<&dyn Optimiser>,
    ) -> OptimisationResults {
        let x0 = match initial {
            Some(v) => v,
            None => self.default_parameters(),
        };

        if let Some(opt) = optimiser {
            return opt.run(self, x0);
        }

        if let Some(default) = &self.default_optimiser {
            return default.run(self, x0);
        }

        // Default to NelderMead when nothing provided
        let nm = NelderMead::new();
        nm.run(self, x0)
    }
}

#[cfg(test)]
mod tests {
    use super::diffsol_problem::test_support::{ConcurrencyProbe, ProbeInstall};
    use super::*;
    use rayon::ThreadPoolBuilder;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    fn build_logistic_problem(backend: DiffsolBackend) -> Problem {
        let dsl = r#"
in = [r, k]
r { 1 }
k { 1 }
u_i { y = 0.1 }
F_i { (r * y) * (1 - (y / k)) }
"#;

        let t_span: Vec<f64> = (0..6).map(|i| i as f64 * 0.2).collect();
        let data_values: Vec<f64> = t_span.iter().map(|t| 0.1 * (*t).exp()).collect();
        let data = DMatrix::from_vec(t_span.len(), 1, data_values);

        let mut params = HashMap::new();
        params.insert("r".to_string(), 1.0);
        params.insert("k".to_string(), 1.0);

        let mut parameter_specs = ParameterSet::new();
        parameter_specs.push(ParameterSpec::new("r", 1.0, None));
        parameter_specs.push(ParameterSpec::new("k", 1.0, None));

        Problem::new_diffsol(
            dsl,
            data,
            t_span,
            DiffsolConfig::default().with_backend(backend),
            parameter_specs,
            Arc::new(SumSquaredError),
            None,
        )
        .expect("failed to build diffsol problem")
    }

    #[test]
    fn diffsol_population_evaluation_matches_individual() {
        let population = vec![
            vec![1.0, 1.0],
            vec![0.9, 1.2],
            vec![1.1, 0.8],
            vec![0.8, 1.3],
        ];

        for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
            let problem = build_logistic_problem(backend);

            let sequential: Vec<f64> = population
                .iter()
                .map(|x| {
                    problem
                        .evaluate(x, false)
                        .expect("sequential evaluation failed")
                })
                .collect();

            let batched: Vec<f64> = problem
                .evaluate_population(&population)
                .into_iter()
                .map(|res| res.expect("batched evaluation failed"))
                .collect();

            assert_eq!(sequential.len(), batched.len());
            for (expected, actual) in sequential.iter().zip(batched.iter()) {
                let diff = (expected - actual).abs();
                assert!(
                    diff <= 1e-8,
                    "[{:?}] expected {}, got {}",
                    backend,
                    expected,
                    actual
                );
            }

            // Assert results are different
            for pair in sequential.windows(2) {
                assert!(pair[0] != pair[1]);
            }
        }
    }

    #[test]
    fn diffsol_population_parallelizes() {
        let population: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let scale = 0.8 + (i as f64) * 0.01;
                vec![1.0 * scale, 1.0 / scale]
            })
            .collect();

        let num_threads = 4;
        for backend in [DiffsolBackend::Dense, DiffsolBackend::Sparse] {
            let problem = build_logistic_problem(backend);

            let probe = ConcurrencyProbe::new(Duration::from_millis(10));
            let _install = ProbeInstall::new(Some(probe.clone()));

            ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .expect("failed to build thread pool")
                .install(|| {
                    let results = problem.evaluate_population(&population);
                    for res in results {
                        res.expect("parallel evaluation failed");
                    }
                });

            let peak = probe.peak();
            assert!(
                peak >= (num_threads / 2),
                "[{:?}] expected peak concurrency at least {}, got {}",
                backend,
                num_threads / 2,
                peak
            );
        }
    }

    #[test]
    fn vector_problem_basic_evaluation() {
        // Simple linear model: y = a * x + b
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let objective = Box::new(|params: &[f64]| -> Result<Vec<f64>, String> {
            let a = params[0];
            let b = params[1];
            Ok((0..5).map(|i| a * (i as f64) + b).collect())
        });

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("a", 1.0, None));
        params.push(ParameterSpec::new("b", 1.0, None));

        let problem = Problem::new_vector(
            objective,
            data,
            vec![5],
            HashMap::new(),
            params,
            Arc::new(SumSquaredError),
            None,
        )
        .expect("failed to create vector problem");

        // Perfect fit should have zero cost (a=1, b=1 gives [1,2,3,4,5])
        let cost = problem
            .evaluate(&[1.0, 1.0], false)
            .expect("evaluation failed");
        assert!(cost.abs() < 1e-10, "expected near-zero cost, got {}", cost);

        // Non-perfect fit should have positive cost
        let cost = problem
            .evaluate(&[0.5, 0.5], false)
            .expect("evaluation failed");
        assert!(cost > 0.0, "expected positive cost, got {}", cost);
    }

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
        let objective = Box::new(move |params: &[f64]| -> Result<Vec<f64>, String> {
            let r = params[0];
            let y0 = params[1];
            Ok(t_span_clone.iter().map(|&t| y0 * (r * t).exp()).collect())
        });

        let mut params = ParameterSet::new();
        params.push(ParameterSpec::new("r", 1.0, Some((0.0, 3.0))));
        params.push(ParameterSpec::new("y0", 1.0, Some((0.0, 5.0))));

        let problem = Problem::new_vector(
            objective,
            data,
            vec![10],
            HashMap::new(),
            params,
            Arc::new(SumSquaredError),
            None,
        )
        .expect("failed to create vector problem");

        // Test with true parameters
        let cost = problem
            .evaluate(&[true_r, true_y0], false)
            .expect("evaluation failed");
        assert!(
            cost.abs() < 1e-10,
            "expected near-zero cost with true params, got {}",
            cost
        );

        // Test with wrong parameters
        let cost = problem
            .evaluate(&[1.0, 1.0], false)
            .expect("evaluation failed");
        assert!(cost > 0.0, "expected positive cost with wrong params");
    }

    #[test]
    fn vector_problem_dimension_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let objective = Box::new(|_params: &[f64]| -> Result<Vec<f64>, String> {
            Ok(vec![1.0, 2.0, 3.0, 4.0, 5.0]) // Wrong size!
        });

        let problem = Problem::new_vector(
            objective,
            data,
            vec![3],
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(SumSquaredError),
            None,
        )
        .expect("failed to create vector problem");

        let result = problem.evaluate(&[1.0], false);
        assert!(result.is_err(), "expected error for dimension mismatch");
        assert!(result.unwrap_err().contains("produced 5 elements but data"));
    }

    #[test]
    fn vector_problem_population_evaluation() {
        let data = vec![1.0, 2.0, 3.0];
        let objective = Box::new(|params: &[f64]| -> Result<Vec<f64>, String> {
            let scale = params[0];
            Ok(vec![scale, 2.0 * scale, 3.0 * scale])
        });

        let problem = Problem::new_vector(
            objective,
            data,
            vec![3],
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(SumSquaredError),
            None,
        )
        .expect("failed to create vector problem");

        let population = vec![vec![1.0], vec![0.5], vec![1.5], vec![2.0]];

        let sequential: Vec<f64> = population
            .iter()
            .map(|x| {
                problem
                    .evaluate(x, false)
                    .expect("sequential evaluation failed")
            })
            .collect();

        let batched: Vec<f64> = problem
            .evaluate_population(&population)
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
        let objective = Box::new(|params: &[f64]| -> Result<Vec<f64>, String> {
            let offset = params[0];
            Ok(vec![1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset])
        });

        let problem = Problem::new_vector(
            objective,
            data,
            vec![4],
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(RootMeanSquaredError),
            None,
        )
        .expect("failed to create vector problem");

        // Perfect fit
        let cost = problem.evaluate(&[0.0], false).expect("evaluation failed");
        assert!(cost.abs() < 1e-10);

        // Offset of 1.0 should give RMSE of 1.0
        let cost = problem.evaluate(&[1.0], false).expect("evaluation failed");
        assert!(
            (cost - 1.0).abs() < 1e-10,
            "expected RMSE of 1.0, got {}",
            cost
        );
    }

    #[test]
    fn vector_problem_builder_pattern() {
        let data = vec![2.0, 4.0, 6.0];
        let objective = Box::new(|params: &[f64]| -> Result<Vec<f64>, String> {
            let scale = params[0];
            Ok(vec![scale * 2.0, scale * 4.0, scale * 6.0])
        });

        let problem = VectorProblemBuilder::new()
            .with_objective(objective)
            .with_data(data)
            .with_parameter(ParameterSpec::new("scale", 1.0, Some((0.0, 10.0))))
            .with_cost_metric(SumSquaredError)
            .build()
            .expect("failed to build vector problem");

        let cost = problem.evaluate(&[1.0], false).expect("evaluation failed");
        assert!(cost.abs() < 1e-10);
    }

    #[test]
    fn vector_problem_empty_data_error() {
        let data = vec![];
        let objective = Box::new(|_params: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![]) });

        let result = Problem::new_vector(
            objective,
            data,
            vec![],
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(SumSquaredError),
            None,
        );

        assert!(result.is_err(), "expected error for empty data");
        let err_msg = result.err().unwrap();
        assert!(err_msg.contains("at least one element"));
    }

    #[test]
    fn vector_problem_shape_validation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let objective =
            Box::new(|_params: &[f64]| -> Result<Vec<f64>, String> { Ok(vec![1.0; 6]) });

        // Valid shape
        let result = Problem::new_vector(
            objective.clone(),
            data.clone(),
            vec![2, 3],
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(SumSquaredError),
            None,
        );
        assert!(result.is_ok(), "expected success with valid shape");

        // Invalid shape
        let result = Problem::new_vector(
            objective,
            data,
            vec![2, 2], // 2*2=4, but data has 6 elements
            HashMap::new(),
            ParameterSet::new(),
            Arc::new(SumSquaredError),
            None,
        );
        assert!(result.is_err(), "expected error for invalid shape");
        let err_msg = result.err().unwrap();
        assert!(err_msg.contains("does not match provided shape"));
    }

    #[test]
    fn scalar_problem_exposes_gradient_from_builder() {
        let problem = ScalarProblemBuilder::new()
            .with_objective_and_gradient(
                |x: &[f64]| x[0] * x[0] + 3.0 * x[0] * x[1] + 2.0 * x[1] * x[1],
                |x: &[f64]| vec![2.0 * x[0] + 3.0 * x[1], 3.0 * x[0] + 4.0 * x[1]],
            )
            .build()
            .expect("failed to build scalar problem");

        let x = [1.0_f64, 2.0_f64];
        let grad = problem
            .gradient()
            .expect("expected gradient to be available for scalar problem");
        let g = grad(&x);

        assert_eq!(g.len(), 2);
        assert!((g[0] - (2.0 * x[0] + 3.0 * x[1])).abs() < 1e-12);
        assert!((g[1] - (3.0 * x[0] + 4.0 * x[1])).abs() < 1e-12);
    }
}
