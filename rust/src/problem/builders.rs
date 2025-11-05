use std::collections::HashMap;
use std::sync::Arc;

use crate::cost::{CostMetric, SumSquaredError};
use crate::optimisers::Optimiser;
use nalgebra::DMatrix;

use super::{CallableObjective, GradientFn, ObjectiveFn, Problem, ProblemKind, SharedOptimiser};

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

#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSpec {
    pub name: String,
    pub initial_value: f64,
    pub bounds: Option<(f64, f64)>,
}

impl ParameterSpec {
    pub fn new<N>(name: N, initial_value: f64, bounds: Option<(f64, f64)>) -> Self
    where
        N: Into<String>,
    {
        Self {
            name: name.into(),
            initial_value,
            bounds,
        }
    }
}

#[derive(Clone, Default)]
pub struct ParameterSet(Vec<ParameterSpec>);

impl ParameterSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn specs(&self) -> &[ParameterSpec] {
        &self.0
    }

    pub fn push(&mut self, spec: ParameterSpec) {
        self.0.push(spec)
    }

    pub fn clear(&mut self) {
        self.0.clear()
    }

    pub fn take(&mut self) -> Vec<ParameterSpec> {
        std::mem::take(&mut self.0)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, ParameterSpec> {
        self.0.iter()
    }
}

pub trait BuilderWithParameters {
    fn parameters_mut(&mut self) -> &mut ParameterSet;
    fn parameters(&self) -> &ParameterSet;
}

/// Extension trait for builders that support parameters
pub trait BuilderParameterExt: BuilderWithParameters + Sized {
    fn with_parameter(mut self, spec: impl Into<ParameterSpec>) -> Self {
        self.parameters_mut().push(spec.into());
        self
    }

    /// Clears previously supplied parameters.
    fn clear_parameters(&mut self) -> &mut Self {
        self.parameters_mut().clear();
        self
    }
}

// Auto-implement for all builders
impl<T: BuilderWithParameters> BuilderParameterExt for T {}

#[derive(Default, Clone)]
pub struct OptimiserSlot(Option<SharedOptimiser>);

impl OptimiserSlot {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set<O>(&mut self, optimiser: O)
    where
        O: Optimiser + Send + Sync + 'static,
    {
        self.0 = Some(Arc::new(optimiser));
    }

    pub fn set_shared(&mut self, optimiser: SharedOptimiser) {
        self.0 = Some(optimiser);
    }

    pub fn clear(&mut self) {
        self.0 = None;
    }

    pub fn take(&mut self) -> Option<SharedOptimiser> {
        self.0.take()
    }

    pub fn get(&self) -> Option<&SharedOptimiser> {
        self.0.as_ref()
    }

    pub fn is_set(&self) -> bool {
        self.0.is_some()
    }
}

// Base trait for accessing the slot (like BuilderWithParameters)
pub trait BuilderWithOptimiser {
    fn optimiser_slot(&mut self) -> &mut OptimiserSlot;
    fn optimiser_slot_ref(&self) -> &OptimiserSlot;
}

// Extension trait with builder methods (like BuilderParameterExt)
pub trait BuilderOptimiserExt: BuilderWithOptimiser + Sized {
    fn with_optimiser<O>(mut self, optimiser: O) -> Self
    where
        O: Optimiser + Send + Sync + 'static,
    {
        self.optimiser_slot().set(optimiser);
        self
    }

    fn with_shared_optimiser(mut self, optimiser: SharedOptimiser) -> Self {
        self.optimiser_slot().set_shared(optimiser);
        self
    }

    fn clear_optimiser(mut self) -> Self {
        self.optimiser_slot().clear();
        self
    }

    fn take_optimiser(&mut self) -> Option<SharedOptimiser> {
        self.optimiser_slot().take()
    }

    fn get_optimiser(&self) -> Option<&SharedOptimiser> {
        self.optimiser_slot_ref().get()
    }

    fn has_optimiser(&self) -> bool {
        self.optimiser_slot_ref().is_set()
    }
}

// Auto-implement for all builders (like parameters)
impl<T: BuilderWithOptimiser> BuilderOptimiserExt for T {}

/// Builder pattern for callable optimisation problems.
pub struct Builder {
    objective: Option<ObjectiveFn>,
    gradient: Option<GradientFn>,
    config: HashMap<String, f64>,
    parameters: ParameterSet,
    optimiser_slot: OptimiserSlot,
}

impl Builder {
    /// Creates a new, empty builder with no objective, gradient, or parameters.
    pub fn new() -> Self {
        Self {
            objective: None,
            gradient: None,
            config: HashMap::new(),
            parameters: ParameterSet::default(),
            optimiser_slot: OptimiserSlot::default(),
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

    /// Finalises the builder, producing a callable optimisation problem.
    pub fn build(mut self) -> Result<Problem, String> {
        let objective = self
            .objective
            .take()
            .ok_or_else(|| "At least one objective must be provide".to_string())?;
        let gradient = self.gradient.take();

        // let parameter_specs = BuilderWithParameters::take_parameters(&mut self);
        let default_optimiser = self.optimiser_slot.take();

        Ok(Problem {
            kind: ProblemKind::Callable(CallableObjective::new(objective, gradient)),
            config: self.config,
            parameter_specs: self.parameters,
            default_optimiser,
        })
    }
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl BuilderWithParameters for Builder {
    fn parameters_mut(&mut self) -> &mut ParameterSet {
        &mut self.parameters
    }

    fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }
}

impl BuilderWithOptimiser for Builder {
    fn optimiser_slot(&mut self) -> &mut OptimiserSlot {
        &mut self.optimiser_slot
    }

    fn optimiser_slot_ref(&self) -> &OptimiserSlot {
        &self.optimiser_slot
    }
}

#[derive(Clone)]
pub struct DiffsolBuilder {
    dsl: Option<String>,
    data: Option<DMatrix<f64>>,
    config: DiffsolConfig,
    parameters: ParameterSet,
    optimiser_slot: OptimiserSlot,
    cost_metric: Arc<dyn CostMetric>,
}

impl DiffsolBuilder {
    /// Creates a new builder with default tolerances and no DiffSL definition or data.
    pub fn new() -> Self {
        Self {
            dsl: None,
            data: None,
            config: DiffsolConfig::default(),
            parameters: ParameterSet::default(),
            optimiser_slot: OptimiserSlot::default(),
            cost_metric: Arc::new(SumSquaredError),
        }
    }

    /// Registers the DiffSL differential equation system.
    pub fn with_diffsl(mut self, dsl: String) -> Self {
        self.dsl = Some(dsl);
        self
    }

    /// Removes any previously registered DiffSL program.
    pub fn remove_diffsl(mut self) -> Self {
        self.dsl = None;
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

    /// Enable or disable parallel evaluation of populations.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.config.parallel = parallel;
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

    /// Finalises the builder into an optimisation problem.
    pub fn build(mut self) -> Result<Problem, String> {
        let dsl = self.dsl.take().ok_or("DSL must be provided")?;
        let data_with_t = self.data.take().ok_or("Data must be provided")?;
        if data_with_t.ncols() < 2 {
            return Err(
                "Data must include at least two columns: t_span followed by observed values"
                    .to_string(),
            );
        }

        let t_span: Vec<f64> = data_with_t.column(0).iter().cloned().collect();
        let data = data_with_t.columns(1, data_with_t.ncols() - 1).into_owned();
        let default_optimiser = self.optimiser_slot.take();

        Problem::new_diffsol(
            &dsl,
            data,
            t_span,
            self.config,
            self.parameters,
            Arc::clone(&self.cost_metric),
            default_optimiser,
        )
    }
}

impl Default for DiffsolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BuilderWithParameters for DiffsolBuilder {
    fn parameters_mut(&mut self) -> &mut ParameterSet {
        &mut self.parameters
    }

    fn parameters(&self) -> &ParameterSet {
        &self.parameters
    }
}

impl BuilderWithOptimiser for DiffsolBuilder {
    fn optimiser_slot(&mut self) -> &mut OptimiserSlot {
        &mut self.optimiser_slot
    }

    fn optimiser_slot_ref(&self) -> &OptimiserSlot {
        &self.optimiser_slot
    }
}
