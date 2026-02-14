use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "stubgen")]
use std::env;

#[cfg(feature = "stubgen")]
use std::path::PathBuf;

use diffid_core::cost::{CostMetric, GaussianNll, RootMeanSquaredError, SumSquaredError};
use diffid_core::prelude::*;
use diffid_core::sampler::SamplingResults;

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};

// Module declarations
mod builders;
mod errors;
mod optimisers;
mod results;
mod samplers;

// Re-exports
pub use builders::{PyDiffsolBuilder, PyScalarBuilder, PyVectorBuilder};
pub use optimisers::{PyAdam, PyAdamState, PyCMAES, PyCMAESState, PyNelderMead, PyNelderMeadState};
pub use results::{PyDone, PyEvaluate, PyOptimisationResults};
pub use samplers::{
    PyDynamicNestedSampler, PyDynamicNestedSamplerState, PyMetropolisHastings,
    PyMetropolisHastingsState, PyNestedSamples, PySamples,
};

use optimisers::Optimiser;
use samplers::Sampler;

type ParameterSpecEntry = (String, f64, Option<(f64, f64)>);

// Import objective types for the problem enum
use diffid_core::problem::{DiffsolObjective, ScalarObjective, VectorObjective};

// Type aliases to reduce complexity warnings
type BoxedScalarFn = Box<dyn Fn(&[f64]) -> f64 + Send + Sync>;
type BoxedGradientFn = Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>;
type ScalarProblemType = Problem<ScalarObjective<BoxedScalarFn>>;
type ScalarGradientProblemType = Problem<ScalarObjective<BoxedScalarFn, BoxedGradientFn>>;

// Enum to hold different Problem types internally
pub(crate) enum DynProblem {
    Scalar(ScalarProblemType),
    ScalarWithGradient(ScalarGradientProblemType),
    Vector(Problem<VectorObjective>),
    Diffsol(Problem<DiffsolObjective>),
}

impl DynProblem {
    fn evaluate(&self, x: &[f64]) -> Result<f64, diffid_core::problem::ProblemError> {
        match self {
            Self::Scalar(p) => p.evaluate(x),
            Self::ScalarWithGradient(p) => p.evaluate(x),
            Self::Vector(p) => p.evaluate(x),
            Self::Diffsol(p) => p.evaluate(x),
        }
    }

    fn dimension(&self) -> usize {
        match self {
            Self::Scalar(p) => p.dimensions(),
            Self::ScalarWithGradient(p) => p.dimensions(),
            Self::Vector(p) => p.dimensions(),
            Self::Diffsol(p) => p.dimensions(),
        }
    }

    fn default_parameters(&self) -> Vec<f64> {
        match self {
            Self::Scalar(p) => p.default_parameters(),
            Self::ScalarWithGradient(p) => p.default_parameters(),
            Self::Vector(p) => p.default_parameters(),
            Self::Diffsol(p) => p.default_parameters(),
        }
    }

    fn bounds(&self) -> Bounds {
        match self {
            Self::Scalar(p) => p.bounds(),
            Self::ScalarWithGradient(p) => p.bounds(),
            Self::Vector(p) => p.bounds(),
            Self::Diffsol(p) => p.bounds(),
        }
    }

    fn initial_values(&self) -> Vec<f64> {
        match self {
            Self::Scalar(p) => p.initial_values(),
            Self::ScalarWithGradient(p) => p.initial_values(),
            Self::Vector(p) => p.initial_values(),
            Self::Diffsol(p) => p.initial_values(),
        }
    }

    fn optimise(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<&Optimiser>,
    ) -> OptimisationResults {
        match (self, optimiser) {
            (Self::Scalar(p), Some(opt)) => p.optimise(initial, Some(&opt.to_core())),
            (Self::Scalar(p), None) => p.optimise(initial, None),

            (Self::ScalarWithGradient(p), Some(opt)) => p.optimise(initial, Some(&opt.to_core())),
            (Self::ScalarWithGradient(p), None) => p.optimise(initial, None),

            (Self::Vector(p), Some(opt)) => p.optimise(initial, Some(&opt.to_core())),
            (Self::Vector(p), None) => p.optimise(initial, None),

            (Self::Diffsol(p), Some(opt)) => p.optimise(initial, Some(&opt.to_core())),
            (Self::Diffsol(p), None) => p.optimise(initial, None),
        }
    }

    fn sample(&self, initial: Option<Vec<f64>>, sampler: Option<&Sampler>) -> SamplingResults {
        match (self, sampler) {
            (Self::Scalar(p), Some(sampler)) => p.sample(initial, Some(&sampler.to_core())),
            (Self::Scalar(p), None) => p.sample(initial, None),

            (Self::ScalarWithGradient(p), Some(sampler)) => {
                p.sample(initial, Some(&sampler.to_core()))
            }
            (Self::ScalarWithGradient(p), None) => p.sample(initial, None),

            (Self::Vector(p), Some(sampler)) => p.sample(initial, Some(&sampler.to_core())),
            (Self::Vector(p), None) => p.sample(initial, None),

            (Self::Diffsol(p), Some(sampler)) => p.sample(initial, Some(&sampler.to_core())),
            (Self::Diffsol(p), None) => p.sample(initial, None),
        }
    }
}

// Cost Metrics
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "CostMetric")]
#[derive(Clone)]
pub struct PyCostMetric {
    inner: Arc<dyn CostMetric>,
    name: &'static str,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyCostMetric {
    /// Name of the cost metric.
    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    fn __repr__(&self) -> String {
        format!("CostMetric(name='{}')", self.name)
    }
}

impl PyCostMetric {
    fn from_metric<M>(metric: M, name: &'static str) -> Self
    where
        M: CostMetric + 'static,
    {
        Self {
            inner: Arc::new(metric),
            name,
        }
    }

    pub(crate) fn metric_arc(&self) -> Arc<dyn CostMetric> {
        Arc::clone(&self.inner)
    }
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "SSE")]
#[pyo3(signature = (weight = 1.0))]
fn sse(weight: f64) -> PyCostMetric {
    PyCostMetric::from_metric(SumSquaredError::new(Some(weight)), "sse")
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "RMSE")]
#[pyo3(signature = (weight = 1.0))]
fn rmse(weight: f64) -> PyCostMetric {
    PyCostMetric::from_metric(RootMeanSquaredError::new(Some(weight)), "rmse")
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "GaussianNLL")]
#[pyo3(signature = (variance, weight = 1.0))]
fn gaussian_nll(variance: f64, weight: f64) -> PyResult<PyCostMetric> {
    let metric = GaussianNll::try_new(Some(weight), variance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(PyCostMetric::from_metric(metric, "gaussian_nll"))
}

// Problem
/// Executable optimisation problem wrapping the Diffid core implementation.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Problem")]
pub struct PyProblem {
    pub(crate) inner: DynProblem,
    pub(crate) default_optimiser: Option<Optimiser>,
    pub(crate) default_sampler: Option<Sampler>,
    pub(crate) parameter_specs: Vec<ParameterSpecEntry>,
    pub(crate) config: HashMap<String, f64>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyProblem {
    /// Evaluate the configured objective function at `x`.
    fn evaluate(&self, x: Vec<f64>) -> PyResult<f64> {
        self.inner.evaluate(&x).map_err(|e| {
            crate::errors::evaluation_error_to_py(diffid_core::errors::EvaluationError::message(
                format!("{e}"),
            ))
        })
    }

    /// Evaluate the gradient of the objective function at `x` if available.
    fn evaluate_gradient(&self, x: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        match &self.inner {
            DynProblem::ScalarWithGradient(p) => match p.evaluate_with_gradient(&x) {
                Ok((_val, grad_opt)) => Ok(grad_opt),
                Err(e) => Err(crate::errors::evaluation_error_to_py(
                    diffid_core::errors::EvaluationError::message(format!("{e}")),
                )),
            },
            _ => Ok(None),
        }
    }

    #[pyo3(signature = (initial=None, optimiser=None))]
    /// Solve the problem starting from `initial` using the supplied optimiser.
    fn optimise(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<Optimiser>,
    ) -> PyResult<PyOptimisationResults> {
        let initial = initial.or_else(|| {
            let defaults = self.inner.default_parameters();
            if defaults.is_empty() {
                None
            } else {
                Some(defaults)
            }
        });
        let opt_ref = optimiser.as_ref().or(self.default_optimiser.as_ref());
        let result = self.inner.optimise(initial, opt_ref);

        Ok(PyOptimisationResults { inner: result })
    }

    #[pyo3(signature = (initial=None, sampler=None))]
    /// Sample from the problem starting from `initial` using the supplied sampler.
    fn sample(
        &self,
        py: Python<'_>,
        initial: Option<Vec<f64>>,
        sampler: Option<Sampler>,
    ) -> PyResult<Py<PyAny>> {
        let initial = initial.or_else(|| {
            let defaults = self.inner.default_parameters();
            if defaults.is_empty() {
                None
            } else {
                Some(defaults)
            }
        });
        let sampler_ref = sampler.as_ref().or(self.default_sampler.as_ref());
        let result = self.inner.sample(initial, sampler_ref);

        match result {
            SamplingResults::MCMC(samples) => {
                Ok(Py::new(py, PySamples { inner: samples })?.into_any())
            }
            SamplingResults::Nested(nested) => {
                Ok(Py::new(py, PyNestedSamples { inner: nested })?.into_any())
            }
        }
    }

    /// Return the numeric configuration value stored under `key` if present.
    fn get_config(&self, _key: String) -> Option<f64> {
        // Config storage has been removed in the refactored API
        None
    }

    /// Return the number of parameters the problem expects.
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    /// Return the parameter bounds for the problem as a list of (lower, upper) tuples.
    fn bounds(&self) -> Vec<(f64, f64)> {
        let bounds = self.inner.bounds();
        bounds
            .limits()
            .iter()
            .map(|r| (*r.start(), *r.end()))
            .collect()
    }

    fn parameters(&self) -> Vec<ParameterSpecEntry> {
        self.parameter_specs.clone()
    }

    fn initial_values(&self) -> Vec<f64> {
        self.inner.initial_values()
    }

    /// Return the default parameter vector implied by the builder.
    #[pyo3(name = "default_parameters")]
    fn default_parameters_py(&self) -> Vec<f64> {
        self.inner.default_parameters()
    }

    /// Return a copy of the problem configuration dictionary.
    fn config(&self) -> HashMap<String, f64> {
        self.config.clone()
    }

    /// Call the problem as a function (shorthand for evaluate).
    ///
    /// Allows using `problem(x)` instead of `problem.evaluate(x)`.
    fn __call__(&self, x: Vec<f64>) -> PyResult<f64> {
        self.evaluate(x)
    }

    /// Return a detailed string representation of the problem.
    fn __repr__(&self) -> String {
        let dim = self.inner.dimension();
        format!("Problem(dimension={dim})")
    }

    /// Return a concise string representation of the problem.
    fn __str__(&self) -> String {
        let dim = self.inner.dimension();
        format!("{dim}-dimensional problem")
    }
}

// Stub generation helpers
#[cfg(feature = "stubgen")]
fn resolve_pyproject_path() -> PathBuf {
    if let Some(root) = env::var_os("MATURIN_WORKSPACE_ROOT") {
        let candidate = PathBuf::from(root).join("pyproject.toml");
        if candidate.exists() {
            return candidate;
        }
    }

    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    let manifest_candidate = manifest_dir.join("pyproject.toml");
    if manifest_candidate.exists() {
        return manifest_candidate;
    }

    manifest_dir
        .parent()
        .map(|parent| parent.join("pyproject.toml"))
        .unwrap_or(manifest_candidate)
}

#[cfg(feature = "stubgen")]
pub fn stub_info() -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    pyo3_stub_gen::StubInfo::from_pyproject_toml(resolve_pyproject_path())
}

#[cfg(feature = "stubgen")]
pub fn stub_info_from(
    pyproject: impl AsRef<std::path::Path>,
) -> pyo3_stub_gen::Result<pyo3_stub_gen::StubInfo> {
    pyo3_stub_gen::StubInfo::from_pyproject_toml(pyproject)
}

// Module Registration
/// Return a convenience factory for creating `Builder` instances.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn builder_factory_py() -> PyScalarBuilder {
    PyScalarBuilder::new()
}

#[pymodule]
fn _diffid(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main classes
    m.add_class::<PyScalarBuilder>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyNelderMead>()?;
    m.add_class::<PyCMAES>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyOptimisationResults>()?;
    m.add_class::<PyDiffsolBuilder>()?;
    m.add_class::<PyVectorBuilder>()?;
    m.add_class::<PyCostMetric>()?;
    m.add_class::<PySamples>()?;
    m.add_class::<PyNestedSamples>()?;
    m.add_class::<PyMetropolisHastings>()?;
    m.add_class::<PyDynamicNestedSampler>()?;
    // Ask-tell types
    m.add_class::<PyEvaluate>()?;
    m.add_class::<PyDone>()?;
    // Optimiser states
    m.add_class::<PyAdamState>()?;
    m.add_class::<PyCMAESState>()?;
    m.add_class::<PyNelderMeadState>()?;
    // Sampler states
    m.add_class::<PyMetropolisHastingsState>()?;
    m.add_class::<PyDynamicNestedSamplerState>()?;

    // Add cost metric factory functions to top level
    m.add_function(wrap_pyfunction!(sse, m)?)?;
    m.add_function(wrap_pyfunction!(rmse, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_nll, m)?)?;

    // Builder submodule
    let builder_module = PyModule::new(py, "builder")?;
    builder_module.add_class::<PyDiffsolBuilder>()?;
    builder_module.add_class::<PyVectorBuilder>()?;
    builder_module.add_class::<PyScalarBuilder>()?;
    // Add aliases with "Problem" naming convention
    let diffsol_type = PyType::new::<PyDiffsolBuilder>(py);
    let vector_type = PyType::new::<PyVectorBuilder>(py);
    let scalar_type = PyType::new::<PyScalarBuilder>(py);
    builder_module.add("DiffsolProblemBuilder", diffsol_type)?;
    builder_module.add("VectorProblemBuilder", vector_type)?;
    builder_module.add("ScalarProblemBuilder", scalar_type)?;
    m.add_submodule(&builder_module)?;
    m.setattr("builder", &builder_module)?;

    let cost_module = PyModule::new(py, "cost")?;
    cost_module.add_class::<PyCostMetric>()?;
    cost_module.add_function(wrap_pyfunction!(sse, &cost_module)?)?;
    cost_module.add_function(wrap_pyfunction!(rmse, &cost_module)?)?;
    cost_module.add_function(wrap_pyfunction!(gaussian_nll, &cost_module)?)?;
    m.add_submodule(&cost_module)?;
    m.setattr("cost", &cost_module)?;

    let sampler_module = PyModule::new(py, "sampler")?;
    sampler_module.add_class::<PyMetropolisHastings>()?;
    sampler_module.add_class::<PyDynamicNestedSampler>()?;
    sampler_module.add_class::<PyNestedSamples>()?;
    sampler_module.add_class::<PySamples>()?;
    m.add_submodule(&sampler_module)?;
    m.setattr("sampler", &sampler_module)?;

    // Register submodules for `import diffid.builder` and `diffid.cost`
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("diffid.builder", &builder_module)?;
    sys_modules.set_item("diffid.cost", &cost_module)?;
    sys_modules.set_item("diffid.sampler", &sampler_module)?;

    // Factory function
    m.add_function(wrap_pyfunction!(builder_factory_py, m)?)?;

    Ok(())
}
