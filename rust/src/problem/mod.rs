use crate::optimisers::{NelderMead, OptimisationResults, Optimiser, CMAES};
use diffsol::OdeBuilder;
use nalgebra::DMatrix;
use std::collections::HashMap;

pub mod diffsol_problem;
pub use diffsol_problem::DiffsolCost;

pub type ObjectiveFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;

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

    pub fn with_backend(mut self, backend: DiffsolBackend) -> Self {
        self.config.backend = backend;
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
            ProblemKind::Callable(obj) => Ok((obj)(x)),
            ProblemKind::Diffsol(cost) => cost.evaluate(x),
        }
    }

    pub fn evaluate_population(&self, xs: &[Vec<f64>]) -> Vec<Result<f64, String>> {
        match &self.kind {
            ProblemKind::Callable(obj) => xs.iter().map(|x| Ok((obj)(x))).collect(),
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
    use std::time::Duration;

    #[test]
    fn diffsol_population_evaluation_matches_individual() {
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

        let problem = Problem::new_diffsol(
            dsl,
            data,
            t_span,
            DiffsolConfig::default(),
            parameter_names,
            params,
        )
        .expect("failed to build diffsol problem");

        let population = vec![
            vec![1.0, 1.0],
            vec![0.9, 1.2],
            vec![1.1, 0.8],
            vec![0.8, 1.3],
        ];

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
            assert!(diff <= 1e-8, "expected {expected}, got {actual}");
        }
    }

    #[test]
    fn diffsol_population_parallelizes() {
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

        let problem = Problem::new_diffsol(
            dsl,
            data,
            t_span,
            DiffsolConfig::default(),
            parameter_names,
            params,
        )
        .expect("failed to build diffsol problem");

        let probe = ConcurrencyProbe::new(Duration::from_millis(10));
        let _install = ProbeInstall::new(Some(probe.clone()));

        let population: Vec<Vec<f64>> = (0..100)
            .map(|i| {
                let scale = 0.8 + (i as f64) * 0.01;
                vec![1.0 * scale, 1.0 / scale]
            })
            .collect();

        let num_threads = 4;
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
            "expected significant concurrency, peak={} with {} threads",
            peak,
            num_threads
        );
    }
}
