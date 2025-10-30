use crate::optimisers::{NelderMead, OptimisationResults, Optimiser, CMAES};
use diffsol::OdeBuilder;
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Arc;

pub mod diffsol_problem;
pub use crate::cost::{CostMetric, RootMeanSquaredError, SumSquaredError};
pub use diffsol_problem::DiffsolCost;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
pub type GradientFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;

struct CallableObjective {
    objective: ObjectiveFn,
    gradient: Option<GradientFn>,
}

impl CallableObjective {
    fn new(objective: ObjectiveFn, gradient: Option<GradientFn>) -> Self {
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

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffsolBackend {
    Dense,
    Sparse,
}

impl Default for DiffsolBackend {
    fn default() -> Self {
        DiffsolBackend::Dense
    }
}

#[derive(Debug, Clone)]
pub struct DiffsolConfig {
    pub rtol: f64,
    pub atol: f64,
    pub backend: DiffsolBackend,
}

impl Default for DiffsolConfig {
    fn default() -> Self {
        Self {
            rtol: DEFAULT_RTOL,
            atol: DEFAULT_ATOL,
            backend: DiffsolBackend::default(),
        }
    }
}

impl DiffsolConfig {
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    pub fn with_backend(mut self, backend: DiffsolBackend) -> Self {
        self.backend = backend;
        self
    }

    pub fn merge(mut self, other: Self) -> Self {
        self.rtol = other.rtol;
        self.atol = other.atol;
        self.backend = other.backend;
        self
    }

    pub fn to_map(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("rtol".to_string(), self.rtol),
            ("atol".to_string(), self.atol),
        ])
    }
}

/// Different kinds of problems
pub enum ProblemKind {
    Callable(CallableObjective),
    Diffsol(Box<DiffsolCost>),
}

/// Builder pattern for the optimisation problem.
pub struct Builder {
    objective: Option<ObjectiveFn>,
    gradient: Option<GradientFn>,
    config: HashMap<String, f64>,
    parameter_names: Vec<String>,
    params: HashMap<String, f64>,
    default_nm: Option<NelderMead>,
    default_cmaes: Option<CMAES>,
}
impl Builder {
    /// Creates a new, empty builder with no objective, gradient, or parameters.
    pub fn new() -> Self {
        Self {
            objective: None,
            gradient: None,
            config: HashMap::new(),
            parameter_names: Vec::new(),
            params: HashMap::new(),
            default_nm: None,
            default_cmaes: None,
        }
    }

    /// Registers an objective callback and clears any previously configured gradient.
    pub fn with_objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.objective = Some(Box::new(f));
        self.gradient = None;
        self
    }

    /// Registers a gradient callback used to compute derivatives of the objective.
    pub fn with_gradient<G>(mut self, g: G) -> Self
    where
        G: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        self.gradient = Some(Box::new(g));
        self
    }

    /// Registers both objective and gradient callbacks in a single call.
    pub fn with_objective_and_gradient<F, G>(mut self, f: F, g: G) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
        G: Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    {
        self.objective = Some(Box::new(f));
        self.gradient = Some(Box::new(g));
        self
    }

    /// Stores an optimisation configuration value keyed by name.
    pub fn with_config(mut self, key: String, value: f64) -> Self {
        self.config.insert(key, value);
        self
    }

    /// Appends a named optimisation parameter preserving insertion order.
    pub fn add_parameter(mut self, name: String) -> Self {
        self.parameter_names.push(name);
        self
    }

    /// Sets Nelder-Mead as the default optimiser, clearing any previous default.
    pub fn set_optimiser_nm(mut self, optimiser: NelderMead) -> Self {
        self.default_nm = Some(optimiser);
        self.default_cmaes = None;
        self
    }

    /// Sets CMA-ES as the default optimiser, clearing any previous default.
    pub fn set_optimiser_cmaes(mut self, optimiser: CMAES) -> Self {
        self.default_cmaes = Some(optimiser);
        self.default_nm = None;
        self
    }

    /// Finalises the builder, producing a callable optimisation problem.
    pub fn build(self) -> Result<Problem, String> {
        match self.objective {
            Some(obj) => Ok(Problem {
                kind: ProblemKind::Callable(CallableObjective::new(obj, self.gradient)),
                config: self.config,
                parameter_names: self.parameter_names,
                params: self.params,
                default_nm: self.default_nm,
                default_cmaes: self.default_cmaes,
            }),
            None => Err("At least one objective must be provide".to_string()),
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for Diffsol problems
#[derive(Clone)]
pub struct DiffsolBuilder {
    dsl: Option<String>,
    data: Option<DMatrix<f64>>,
    config: DiffsolConfig,
    params: HashMap<String, f64>,
    parameter_names: Vec<String>,
    cost_metric: Arc<dyn CostMetric>,
}

impl DiffsolBuilder {
    /// Creates a new builder with default tolerances and no DiffSL definition or data.
    pub fn new() -> Self {
        Self {
            dsl: None,
            data: None,
            config: DiffsolConfig::default(),
            params: HashMap::new(),
            parameter_names: Vec::new(),
            cost_metric: Arc::new(SumSquaredError::default()),
        }
    }

    /// Registers the DiffSL differential equation system.
    pub fn add_diffsl(mut self, dsl: String) -> Self {
        self.dsl = Some(dsl);
        self
    }

    /// Removes any previously registered DiffSL program.
    pub fn remove_diffsl(mut self) -> Self {
        self.dsl = None;
        self
    }

    /// Supplies observed data used to fit the differential model.
    pub fn add_data(mut self, data: DMatrix<f64>) -> Self {
        self.data = Some(data);
        self
    }

    /// Removes any previously supplied observed data and associated time span.
    pub fn remove_data(mut self) -> Self {
        self.data = None;
        self
    }

    /// Sets the relative tolerance applied during integration.
    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.config.rtol = rtol;
        self
    }

    /// Sets the absolute tolerance applied during integration.
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.config.atol = atol;
        self
    }

    /// Chooses the backend implementation (dense or sparse) for the solver.
    pub fn with_backend(mut self, backend: DiffsolBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Selects the cost metric used to compare model outputs against observed data.
    pub fn with_cost_metric<M>(mut self, cost_metric: M) -> Self
    where
        M: CostMetric + 'static,
    {
        self.cost_metric = Arc::new(cost_metric);
        self
    }

    /// Directly set the cost metric from a trait object.
    pub fn with_cost_metric_arc(mut self, cost_metric: Arc<dyn CostMetric>) -> Self {
        self.cost_metric = cost_metric;
        self
    }

    /// Resets the cost metric to the default sum of squared errors.
    pub fn remove_cost(mut self) -> Self {
        self.cost_metric = Arc::new(SumSquaredError::default());
        self
    }

    /// Merges configuration values by name, updating tolerances when provided.
    pub fn add_config(mut self, config: HashMap<String, f64>) -> Self {
        for (key, value) in config {
            match key.as_str() {
                "rtol" => self.config.rtol = value,
                "atol" => self.config.atol = value,
                _ => {}
            }
        }
        self
    }

    /// Supplies named parameter defaults used when solving the DiffSL problem.
    pub fn add_params(mut self, params: HashMap<String, f64>) -> Self {
        // Extract parameter names in insertion order before moving params
        self.parameter_names = params.keys().cloned().collect();
        self.params = params;
        self
    }

    /// Removes all previously supplied parameters and names.
    pub fn remove_params(mut self) -> Self {
        self.parameter_names.clear();
        self.params.clear();
        self
    }

    /// Finalises the builder into a differential-equation optimisation problem.
    pub fn build(self) -> Result<Problem, String> {
        let dsl = self.dsl.ok_or("DSL must be provided")?;
        let data_with_t = self.data.ok_or("Data must be provided")?;
        if data_with_t.ncols() < 2 {
            return Err(
                "Data must include at least two columns: t_span followed by observed values"
                    .to_string(),
            );
        }

        let t_span: Vec<f64> = data_with_t.column(0).iter().cloned().collect();
        let data = data_with_t.columns(1, data_with_t.ncols() - 1).into_owned();

        Problem::new_diffsol(
            &dsl,
            data,
            t_span,
            self.config,
            self.parameter_names,
            self.params,
            Arc::clone(&self.cost_metric),
        )
    }
}

impl Default for DiffsolBuilder {
    fn default() -> Self {
        Self::new()
    }
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
                .map(diffsol_problem::BackendProblem::Dense),
            DiffsolBackend::Sparse => OdeBuilder::<diffsol::FaerSparseMat<f64>>::new()
                .atol([config.atol])
                .rtol(config.rtol)
                .build_from_diffsl(dsl)
                .map_err(|e| format!("Failed to build ODE model: {}", e))
                .map(diffsol_problem::BackendProblem::Sparse),
        }?;

        let cost = DiffsolCost::new(
            backend_problem,
            dsl.to_string(),
            config.clone(),
            data,
            t_span,
            cost_metric,
        );

        Ok(Problem {
            kind: ProblemKind::Diffsol(Box::new(cost)),
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
            ProblemKind::Diffsol(cost) => cost.evaluate(x),
        }
    }

    pub fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, String>> {
        match &self.kind {
            ProblemKind::Callable(callable) => {
                xs.iter().map(|x| Ok(callable.evaluate(x))).collect()
            }
            ProblemKind::Diffsol(cost) => {
                let slices: Vec<&[f64]> = xs.iter().map(|x| x.as_slice()).collect();
                cost.evaluate_population(&slices)
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
            Arc::new(SumSquaredError::default()),
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
