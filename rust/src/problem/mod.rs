use crate::optimisers::{NelderMead, OptimisationResults, Optimiser, CMAES};
use diffsol::OdeBuilder;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Arc;

pub mod builders;
pub mod diffsol_problem;
pub use crate::cost::{CostMetric, RootMeanSquaredError, SumSquaredError};
pub use builders::{Builder, DiffsolBackend, DiffsolBuilder, DiffsolConfig};
pub use diffsol_problem::DiffsolProblem;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type GradientFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

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

/// Different kinds of problems
pub enum ProblemKind {
    Callable(CallableObjective),
    Diffsol(Box<DiffsolProblem>),
}
// Problem class
pub struct Problem {
    kind: ProblemKind,
    config: HashMap<String, f64>,
    parameter_names: Vec<String>,
    params: HashMap<String, f64>,
    default_nm: Option<NelderMead>,
    default_cmaes: Option<CMAES>,
}

impl Problem {
    pub fn new_diffsol(
        dsl: &str,
        data: DMatrix<f64>,
        t_span: Vec<f64>,
        config: DiffsolConfig,
        parameter_names: Vec<String>,
        params: HashMap<String, f64>,
        cost_metric: Arc<dyn CostMetric>,
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
            parameter_names,
            params,
            default_nm: None,
            default_cmaes: None,
        })
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<f64, String> {
        match &self.kind {
            ProblemKind::Callable(callable) => Ok(callable.evaluate(x)),
            ProblemKind::Diffsol(problem) => problem.evaluate(x),
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
        }
    }

    pub fn get_config(&self, key: &str) -> Option<&f64> {
        self.config.get(key)
    }

    pub fn config(&self) -> &HashMap<String, f64> {
        &self.config
    }

    pub fn params(&self) -> &HashMap<String, f64> {
        &self.params
    }

    pub fn dimension(&self) -> usize {
        if !self.parameter_names.is_empty() {
            return self.parameter_names.len();
        }
        0
    }

    pub fn gradient(&self) -> Option<&GradientFn> {
        match &self.kind {
            ProblemKind::Callable(callable) => callable.gradient(),
            ProblemKind::Diffsol(_) => None,
        }
    }

    pub fn optimize(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<&dyn Optimiser>,
    ) -> OptimisationResults {
        let x0 = match initial {
            Some(v) => v,
            None => vec![0.0; self.dimension()],
        };

        if let Some(opt) = optimiser {
            return opt.run(self, x0);
        }

        if let Some(default_nm) = &self.default_nm {
            return default_nm.run(self, x0);
        }

        if let Some(default_cmaes) = &self.default_cmaes {
            return default_cmaes.run(self, x0);
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

        let parameter_names = vec!["r".to_string(), "k".to_string()];

        Problem::new_diffsol(
            dsl,
            data,
            t_span,
            DiffsolConfig::default().with_backend(backend),
            parameter_names,
            params,
            Arc::new(SumSquaredError),
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
                .map(|x| problem.evaluate(x).expect("sequential evaluation failed"))
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
}
