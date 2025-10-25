use crate::optimisers::{NelderMead, OptimisationResults, Optimiser, CMAES};
use diffsol::OdeBuilder;
use nalgebra::DMatrix;
use std::collections::HashMap;

pub mod diffsol_problem;
pub use diffsol_problem::DiffsolCost;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
type M = diffsol::NalgebraMat<f64>;

const DEFAULT_RTOL: f64 = 1e-6;
const DEFAULT_ATOL: f64 = 1e-9;

#[derive(Debug, Clone)]
pub struct DiffsolConfig {
    pub rtol: f64,
    pub atol: f64,
}

impl Default for DiffsolConfig {
    fn default() -> Self {
        Self {
            rtol: DEFAULT_RTOL,
            atol: DEFAULT_ATOL,
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

    pub fn merge(mut self, other: Self) -> Self {
        self.rtol = other.rtol;
        self.atol = other.atol;
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
    Callable(ObjectiveFn),
    Diffsol(Box<DiffsolCost>),
}

// Builder pattern for the optimisation problem
pub struct Builder {
    objective: Option<ObjectiveFn>,
    config: HashMap<String, f64>,
    parameter_names: Vec<String>,
    params: HashMap<String, f64>,
    default_nm: Option<NelderMead>,
    default_cmaes: Option<CMAES>,
}
impl Builder {
    pub fn new() -> Self {
        Self {
            objective: None,
            config: HashMap::new(),
            parameter_names: Vec::new(),
            params: HashMap::new(),
            default_nm: None,
            default_cmaes: None,
        }
    }

    pub fn with_objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&[f64]) -> f64 + Send + Sync + 'static,
    {
        self.objective = Some(Box::new(f));
        self
    }

    pub fn with_config(mut self, key: String, value: f64) -> Self {
        self.config.insert(key, value);
        self
    }

    pub fn add_parameter(mut self, name: String) -> Self {
        self.parameter_names.push(name);
        self
    }

    pub fn set_optimiser_nm(mut self, optimiser: NelderMead) -> Self {
        self.default_nm = Some(optimiser);
        self.default_cmaes = None;
        self
    }

    pub fn set_optimiser_cmaes(mut self, optimiser: CMAES) -> Self {
        self.default_cmaes = Some(optimiser);
        self.default_nm = None;
        self
    }

    pub fn build(self) -> Result<Problem, String> {
        match self.objective {
            Some(obj) => Ok(Problem {
                kind: ProblemKind::Callable(obj),
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
pub struct DiffsolBuilder {
    dsl: Option<String>,
    data: Option<DMatrix<f64>>,
    t_span: Option<Vec<f64>>,
    config: DiffsolConfig,
    params: HashMap<String, f64>,
    parameter_names: Vec<String>,
}

impl DiffsolBuilder {
    pub fn new() -> Self {
        Self {
            dsl: None,
            data: None,
            t_span: None,
            config: DiffsolConfig::default(),
            params: HashMap::new(),
            parameter_names: Vec::new(),
        }
    }

    pub fn add_diffsl(mut self, dsl: String) -> Self {
        self.dsl = Some(dsl);
        self
    }

    pub fn add_data(mut self, data: DMatrix<f64>) -> Self {
        self.data = Some(data);
        self
    }

    pub fn with_t_span(mut self, t_span: Vec<f64>) -> Self {
        self.t_span = Some(t_span);
        self
    }

    pub fn with_rtol(mut self, rtol: f64) -> Self {
        self.config.rtol = rtol;
        self
    }

    pub fn with_atol(mut self, atol: f64) -> Self {
        self.config.atol = atol;
        self
    }

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

    pub fn add_params(mut self, params: HashMap<String, f64>) -> Self {
        // Extract parameter names in insertion order before moving params
        self.parameter_names = params.keys().cloned().collect();
        self.params = params;
        self
    }

    pub fn build(self) -> Result<Problem, String> {
        let dsl = self.dsl.ok_or("DSL must be provided")?;
        let data = self.data.ok_or("Data must be provided")?;
        let t_span = self
            .t_span
            .unwrap_or_else(|| (0..data.len()).map(|i| i as f64).collect());

        Problem::new_diffsol(
            &dsl,
            data,
            t_span,
            self.config,
            self.parameter_names,
            self.params,
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
    ) -> Result<Self, String> {
        let builder = OdeBuilder::<M>::new()
            .atol([config.atol])
            .rtol(config.rtol);
        let model = builder
            .build_from_diffsl(dsl)
            .map_err(|e| format!("Failed to build ODE model: {}", e))?;
        let cost = DiffsolCost::new(model, data, t_span);

        Ok(Problem {
            kind: ProblemKind::Diffsol(Box::new(cost)),
            config: HashMap::new(),
            parameter_names,
            params,
            default_nm: None,
            default_cmaes: None,
        })
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<f64, String> {
        match &self.kind {
            ProblemKind::Callable(obj) => Ok((obj)(x)),
            ProblemKind::Diffsol(cost) => cost.evaluate(x),
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
