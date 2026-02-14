use super::{ParameterSet, ProblemBuilderError};
use crate::cost::{CostMetric, SumSquaredError};
use crate::prelude::{Optimiser, ParameterSpec};
use crate::problem::{DiffsolObjective, ParameterRange, Problem};
use nalgebra::DMatrix;
use std::collections::HashMap;
use std::sync::Arc;

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-8;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DiffsolBackend {
    #[default]
    Dense,
    Sparse,
}

#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct DiffsolConfig {
    pub rtol: f64,
    pub atol: f64,
    pub backend: DiffsolBackend,
    pub parallel: bool,
}

impl Default for DiffsolConfig {
    fn default() -> Self {
        Self {
            rtol: DEFAULT_RTOL,
            atol: DEFAULT_ATOL,
            backend: DiffsolBackend::default(),
            parallel: true,
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

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn merge(mut self, other: Self) -> Self {
        self.rtol = other.rtol;
        self.atol = other.atol;
        self.backend = other.backend;
        self.parallel = other.parallel;
        self
    }

    pub fn to_map(&self) -> HashMap<String, f64> {
        HashMap::from([
            ("rtol".to_string(), self.rtol),
            ("atol".to_string(), self.atol),
            (
                "parallel".to_string(),
                if self.parallel { 1.0 } else { 0.0 },
            ),
        ])
    }
}

#[must_use]
#[derive(Clone)]
pub struct DiffsolProblemBuilder {
    equations: Option<String>,
    data: Option<DMatrix<f64>>,
    costs: Vec<Arc<dyn CostMetric>>,
    config: DiffsolConfig,
    parameters: ParameterSet,
    optimiser: Optimiser,
}

impl Default for DiffsolProblemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl DiffsolProblemBuilder {
    pub fn new() -> Self {
        Self {
            equations: None,
            data: None,
            costs: Vec::new(),
            config: DiffsolConfig::default(),
            parameters: ParameterSet::default(),
            optimiser: Optimiser::default(),
        }
    }

    pub fn with_optimiser(mut self, opt: impl Into<Optimiser>) -> Self {
        self.optimiser = opt.into();
        self
    }

    /// Registers the `DiffSL` differential equation system.
    pub fn with_diffsl(mut self, equations: String) -> Self {
        self.equations = Some(equations);
        self
    }

    /// Add a parameter to the problem
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        initial: f64,
        range: impl Into<ParameterRange>,
    ) -> Self {
        self.parameters
            .push(ParameterSpec::new(name, initial, range));
        self
    }

    /// Supplies observed data used to fit the differential model.
    pub fn with_data(mut self, data: DMatrix<f64>) -> Self {
        self.data = Some(data);
        self
    }

    /// Removes any previously supplied observed data and associated time span.
    pub fn remove_data(mut self) -> Self {
        self.data = None;
        self
    }

    /// Sets the relative and absolute tolerances applied during integration.
    pub fn with_tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.config.rtol = rtol;
        self.config.atol = atol;
        self
    }

    /// Chooses the backend implementation (dense or sparse) for the solver.
    pub fn with_backend(mut self, backend: DiffsolBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Enable or disable parallel evaluation of populations.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
        self
    }

    /// Adds a cost used to compare model outputs against observed data.
    pub fn with_cost<M>(mut self, cost: M) -> Self
    where
        M: CostMetric + 'static,
    {
        self.costs.push(Arc::new(cost));
        self
    }

    /// Directly add the cost metric from a trait object.
    pub fn with_cost_arc(mut self, cost: Arc<dyn CostMetric>) -> Self {
        self.costs.push(cost);
        self
    }

    /// Resets the cost metric to the default sum of squared errors.
    pub fn remove_costs(mut self) -> Self {
        self.costs.clear();
        self
    }

    /// Merges configuration values by name, updating tolerances when provided.
    pub fn with_config(mut self, config: HashMap<String, f64>) -> Self {
        for (key, value) in config {
            match key.as_str() {
                "rtol" => self.config.rtol = value,
                "atol" => self.config.atol = value,
                "parallel" => self.config.parallel = value != 0.0,
                _ => {}
            }
        }
        self
    }

    /// Build the problem
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No data has been provided via `with_data`
    /// - Data has fewer than 2 columns (needs time + at least one observation)
    /// - No differential equations have been provided via `with_diffsl`
    pub fn build(self) -> Result<Problem<DiffsolObjective>, ProblemBuilderError> {
        // Unpack data and verify
        let data_with_t = self.data.as_ref().ok_or(ProblemBuilderError::MissingData)?;
        if data_with_t.ncols() < 2 {
            return Err(ProblemBuilderError::DimensionMismatch {
                expected: 2,
                got: data_with_t.ncols(),
            });
        }
        let t_span: Vec<f64> = data_with_t.column(0).iter().copied().collect();
        let data = data_with_t.columns(1, data_with_t.ncols() - 1).into_owned();

        // Check costs and provide default if empty
        let mut costs = self.costs;
        if costs.is_empty() {
            costs.push(Arc::new(SumSquaredError::default()));
        }

        // Diffsl equation definition
        let equations = self.equations.ok_or(ProblemBuilderError::MissingSystem)?;

        // Build objective
        let objective = DiffsolObjective::new(equations, t_span, data, self.config, costs);

        // Build problem
        Ok(Problem::new(objective, self.parameters).with_optimiser(self.optimiser))
    }
}
