use std::collections::HashMap;
use std::sync::Arc;

use crate::cost::{CostMetric, SumSquaredError};
use crate::optimisers::{NelderMead, CMAES};
use nalgebra::DMatrix;

use super::{CallableObjective, GradientFn, ObjectiveFn, Problem, ProblemKind};

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DiffsolBackend {
    #[default]
    Dense,
    Sparse,
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

/// Builder pattern for callable optimisation problems.
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
            cost_metric: Arc::new(SumSquaredError),
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
        self.cost_metric = Arc::new(SumSquaredError);
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

    /// Finalises the builder into an optimisation problem.
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
