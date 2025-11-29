use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "stubgen")]
use std::env;

#[cfg(feature = "stubgen")]
use std::path::PathBuf;

use chronopt_core::cost::{CostMetric, GaussianNll, RootMeanSquaredError, SumSquaredError};
use chronopt_core::prelude::*;
use chronopt_core::problem::{
    DiffsolBackend, DiffsolProblemBuilder, ScalarProblemBuilder, VectorProblemBuilder,
};
use chronopt_core::sampler::{
    DynamicNestedSampler as CoreDynamicNestedSampler, MetropolisHastings as CoreMetropolisHastings,
    NestedSamples as CoreNestedSamples, Samples as CoreSamples,
};

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::{
    derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods},
    TypeInfo,
};

type ParameterSpecEntry = (String, f64, Option<(f64, f64)>);

// Helper function to convert numpy arrays to DMatrix
fn convert_array_to_dmatrix(data: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<DMatrix<f64>> {
    let array = data.as_array();
    let array_2d = array
        .into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| PyValueError::new_err("Data array must be two-dimensional"))?;

    let (nrows, ncols) = array_2d.dim();
    let mut column_major = Vec::with_capacity(nrows * ncols);

    // nalgebra uses column-major storage
    for col in 0..ncols {
        for row in 0..nrows {
            column_major.push(array_2d[[row, col]]);
        }
    }

    Ok(DMatrix::from_vec(nrows, ncols, column_major))
}

#[cfg(feature = "stubgen")]
pyo3_stub_gen::impl_stub_type!(Optimiser = PyNelderMead | PyCMAES | PyAdam);

// ============================================================================
// Optimiser Enum for Polymorphic Types
// ============================================================================

#[derive(Clone)]
enum Optimiser {
    NelderMead(NelderMead),
    Cmaes(CMAES),
    Adam(Adam),
}

// ============================================================================
// Samplers
// ============================================================================

/// Container for sampler draws and diagnostics.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "Samples")]
pub struct PySamples {
    inner: CoreSamples,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PySamples {
    #[getter]
    fn chains(&self) -> Vec<Vec<Vec<f64>>> {
        self.inner.chains().to_vec()
    }

    #[getter]
    fn mean_x(&self) -> Vec<f64> {
        self.inner.mean_x().to_vec()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draws()
    }

    #[getter]
    fn time(&self) -> Duration {
        self.inner.time()
    }

    fn __repr__(&self) -> String {
        format!(
            "Samples(draws={}, mean_x={:?}, chains={}, time={:?})",
            self.inner.draws(),
            self.inner.mean_x(),
            self.inner.chains().len(),
            self.inner.time()
        )
    }
}

/// Nested sampling results including evidence estimates.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "NestedSamples")]
#[derive(Clone)]
pub struct PyNestedSamples {
    inner: CoreNestedSamples,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyNestedSamples {
    #[getter]
    fn posterior(&self) -> Vec<(Vec<f64>, f64, f64)> {
        self.inner
            .posterior()
            .iter()
            .map(|sample| {
                (
                    sample.position.clone(),
                    sample.log_likelihood,
                    sample.log_weight,
                )
            })
            .collect()
    }

    #[getter]
    fn mean(&self) -> Vec<f64> {
        self.inner.mean().to_vec()
    }

    #[getter]
    fn draws(&self) -> usize {
        self.inner.draws()
    }

    #[getter]
    fn log_evidence(&self) -> f64 {
        self.inner.log_evidence()
    }

    #[getter]
    fn information(&self) -> f64 {
        self.inner.information()
    }

    #[getter]
    fn time(&self) -> Duration {
        self.inner.time()
    }

    fn to_samples(&self) -> PySamples {
        PySamples {
            inner: self.inner.to_samples(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NestedSamples(draws={}, log_evidence={:.3}, information={:.3})",
            self.inner.draws(),
            self.inner.log_evidence(),
            self.inner.information()
        )
    }
}

/// Basic Metropolis-Hastings sampler binding mirroring the optimiser API.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "MetropolisHastings")]
#[derive(Clone)]
pub struct PyMetropolisHastings {
    inner: CoreMetropolisHastings,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyMetropolisHastings {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreMetropolisHastings::new(),
        }
    }

    fn with_num_chains(mut slf: PyRefMut<'_, Self>, num_chains: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_num_chains(num_chains);
        slf
    }

    #[pyo3(name = "set_number_of_chains")]
    fn set_number_of_chains(slf: PyRefMut<'_, Self>, num_chains: usize) -> PyRefMut<'_, Self> {
        Self::with_num_chains(slf, num_chains)
    }

    fn with_iterations(mut slf: PyRefMut<'_, Self>, iterations: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_iterations(iterations);
        slf
    }

    #[pyo3(name = "with_num_steps")]
    fn with_num_steps(slf: PyRefMut<'_, Self>, steps: usize) -> PyRefMut<'_, Self> {
        Self::with_iterations(slf, steps)
    }

    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PySamples {
        let samples = self.inner.run(&problem.inner, initial);
        PySamples { inner: samples }
    }
}

/// Dynamic nested sampler binding exposing DNS configuration knobs.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(module = "chronopt.sampler", name = "DynamicNestedSampler")]
#[derive(Clone)]
pub struct PyDynamicNestedSampler {
    inner: CoreDynamicNestedSampler,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDynamicNestedSampler {
    #[new]
    fn new() -> Self {
        Self {
            inner: CoreDynamicNestedSampler::new(),
        }
    }

    fn with_live_points(mut slf: PyRefMut<'_, Self>, live_points: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_live_points(live_points);
        slf
    }

    fn with_expansion_factor(
        mut slf: PyRefMut<'_, Self>,
        expansion_factor: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_expansion_factor(expansion_factor);
        slf
    }

    fn with_termination_tolerance(
        mut slf: PyRefMut<'_, Self>,
        tolerance: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_termination_tolerance(tolerance);
        slf
    }

    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    #[pyo3(signature = (problem, initial=None))]
    fn run(&self, problem: &PyProblem, initial: Option<Vec<f64>>) -> PyNestedSamples {
        let initial = initial.unwrap_or_else(|| problem.inner.default_parameters());
        let nested = self.inner.run_nested(&problem.inner, initial);
        PyNestedSamples { inner: nested }
    }
}

#[cfg(feature = "stubgen")]
#[allow(dead_code)]
fn optimiser_type_info() -> TypeInfo {
    TypeInfo::unqualified("chronopt._chronopt.NelderMead")
        | TypeInfo::unqualified("chronopt._chronopt.CMAES")
        | TypeInfo::unqualified("chronopt._chronopt.Adam")
}

impl FromPyObject<'_, '_> for Optimiser {
    type Error = PyErr;

    fn extract(obj: Borrowed<'_, '_, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(nm) = obj.extract::<PyRef<PyNelderMead>>() {
            Ok(Optimiser::NelderMead((*nm).inner.clone()))
        } else if let Ok(cma) = obj.extract::<PyRef<PyCMAES>>() {
            Ok(Optimiser::Cmaes((*cma).inner.clone()))
        } else if let Ok(adam) = obj.extract::<PyRef<PyAdam>>() {
            Ok(Optimiser::Adam((*adam).inner.clone()))
        } else {
            Err(PyTypeError::new_err(
                "Optimiser must be an instance of NelderMead, CMAES, or Adam",
            ))
        }
    }
}

// ============================================================================
// Cost Metrics
// ============================================================================

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

    fn metric_arc(&self) -> Arc<dyn CostMetric> {
        Arc::clone(&self.inner)
    }
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "SSE")]
fn sse() -> PyCostMetric {
    PyCostMetric::from_metric(SumSquaredError, "sse")
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "RMSE")]
fn rmse() -> PyCostMetric {
    PyCostMetric::from_metric(RootMeanSquaredError, "rmse")
}

#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction(name = "GaussianNLL")]
#[pyo3(signature = (variance = 1.0))]
fn gaussian_nll(variance: f64) -> PyResult<PyCostMetric> {
    if !variance.is_finite() || variance <= 0.0 {
        return Err(PyValueError::new_err(
            "variance must be positive and finite",
        ));
    }
    Ok(PyCostMetric::from_metric(
        GaussianNll::new(variance),
        "gaussian_nll",
    ))
}

// ============================================================================
// Python Objective Function Wrapper
// ============================================================================

struct PyObjectiveFn {
    callable: Py<PyAny>,
}

impl PyObjectiveFn {
    fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    fn call(&self, x: &[f64]) -> PyResult<f64> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            if let Ok(output) = result.extract::<PyReadonlyArray1<f64>>() {
                let array = output.as_array();
                return match array.len() {
                    1 => Ok(array[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective array must contain exactly one element, got {}",
                        n
                    ))),
                };
            }

            if let Ok(values) = result.extract::<Vec<f64>>() {
                return match values.len() {
                    1 => Ok(values[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective sequence must contain exactly one element, got {}",
                        n
                    ))),
                };
            }

            if let Ok(value) = result.extract::<f64>() {
                return Ok(value);
            }

            let ty_name = result
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "unknown".to_string());

            Err(PyTypeError::new_err(format!(
                "Objective callable must return a float, numpy array, or single-element sequence; got {}",
                ty_name
            )))
        })
    }
}

struct PyGradientFn {
    callable: Py<PyAny>,
}

impl PyGradientFn {
    fn new(callable: Py<PyAny>) -> Self {
        Self { callable }
    }

    fn call(&self, x: &[f64]) -> PyResult<Vec<f64>> {
        Python::attach(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            if let Ok(output) = result.extract::<PyReadonlyArray1<f64>>() {
                return Ok(output.as_array().to_vec());
            }

            result.extract::<Vec<f64>>()
        })
    }
}

// ============================================================================
// Builder
// ============================================================================

/// High-level builder for optimisation `Problem` instances exposed to Python.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "ScalarBuilder")]
pub struct PyScalarBuilder {
    inner: ScalarProblemBuilder,
    py_callable: Option<Arc<PyObjectiveFn>>,
    py_gradient: Option<Arc<PyGradientFn>>,
    default_optimiser: Option<Optimiser>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyScalarBuilder {
    /// Create an empty builder with no objective, parameters, or default optimiser.
    #[new]
    fn new() -> Self {
        Self {
            inner: ScalarProblemBuilder::new(),
            py_callable: None,
            py_gradient: None,
            default_optimiser: None,
        }
    }

    /// Configure the default optimiser used when `Problem.optimize` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        match &optimiser {
            Optimiser::NelderMead(nm) => {
                slf.inner = std::mem::take(&mut slf.inner).with_optimiser(nm.clone());
            }
            Optimiser::Cmaes(cma) => {
                slf.inner = std::mem::take(&mut slf.inner).with_optimiser(cma.clone());
            }
            Optimiser::Adam(adam) => {
                slf.inner = std::mem::take(&mut slf.inner).with_optimiser(adam.clone());
            }
        }

        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Attach the objective function callable executed during optimisation.
    fn with_callable(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        Python::attach(|py| {
            if !obj.bind(py).is_callable() {
                return Err(PyTypeError::new_err("Object must be callable"));
            }
            Ok(())
        })?;

        let py_fn = Arc::new(PyObjectiveFn::new(obj));
        let objective = Arc::clone(&py_fn);

        slf.inner = std::mem::take(&mut slf.inner)
            .with_objective(move |x: &[f64]| objective.call(x).unwrap_or(f64::INFINITY));
        slf.py_callable = Some(py_fn);
        Ok(slf)
    }

    /// Attach the gradient callable returning derivatives of the objective.
    fn with_gradient(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        Python::attach(|py| {
            if !obj.bind(py).is_callable() {
                return Err(PyTypeError::new_err("Object must be callable"));
            }
            Ok(())
        })?;

        let py_grad = Arc::new(PyGradientFn::new(obj));
        let grad = Arc::clone(&py_grad);

        slf.inner = std::mem::take(&mut slf.inner).with_gradient(move |x: &[f64]| {
            grad.call(x).unwrap_or_else(|_| vec![f64::NAN; x.len()])
        });
        slf.py_gradient = Some(py_grad);
        Ok(slf)
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyRefMut<'_, Self> {
        let spec = ParameterSpec::new(name.clone(), initial_value, bounds);
        slf.inner = std::mem::take(&mut slf.inner).with_parameter(spec);
        slf
    }

    /// Finalize the builder into an executable `Problem`.
    fn build(&mut self) -> PyResult<PyProblem> {
        let problem = self.inner.build().map_err(PyValueError::new_err)?;
        Ok(PyProblem {
            inner: problem,
            default_optimiser: self.default_optimiser.clone(),
        })
    }
}

// ============================================================================
// Diffsol Builder
// ============================================================================

/// Differential equation solver builder.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "DiffsolBuilder")]
pub struct PyDiffsolBuilder {
    inner: DiffsolProblemBuilder,
    default_optimiser: Option<Optimiser>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyDiffsolBuilder {
    /// Create an empty differential solver builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: DiffsolProblemBuilder::new(),
            default_optimiser: None,
        }
    }

    fn __copy__(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            default_optimiser: self.default_optimiser.clone(),
        }
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyDict>) -> Self {
        self.__copy__()
    }

    /// Register the DiffSL program describing the system dynamics.
    fn with_diffsl(mut slf: PyRefMut<'_, Self>, dsl: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_diffsl(dsl);
        slf
    }

    /// Remove any registered DiffSL program.
    fn remove_diffsl(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_diffsl();
        slf
    }

    /// Attach observed data used to fit the differential equation.
    ///
    /// The first column must contain the time samples (t_span) and the remaining
    /// columns the observed trajectories.
    fn with_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_matrix = convert_array_to_dmatrix(&data)?;
        if data_matrix.ncols() < 2 {
            return Err(PyValueError::new_err(
                "Data must include at least two columns with t_span in the first column",
            ));
        }
        slf.inner = std::mem::take(&mut slf.inner).with_data(data_matrix);
        Ok(slf)
    }

    /// Remove any previously attached data along with its time span.
    fn remove_data(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_data();
        slf
    }

    /// Choose whether to use dense or sparse diffusion solvers.
    fn with_backend(mut slf: PyRefMut<'_, Self>, backend: String) -> PyResult<PyRefMut<'_, Self>> {
        let backend_enum = match backend.as_str() {
            "dense" => DiffsolBackend::Dense,
            "sparse" => DiffsolBackend::Sparse,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown backend '{}'. Expected 'dense' or 'sparse'",
                    other
                )))
            }
        };
        slf.inner = std::mem::take(&mut slf.inner).with_backend(backend_enum);
        Ok(slf)
    }

    /// Opt into parallel proposal generation when supported by the backend.
    #[pyo3(signature = (parallel=None))]
    fn with_parallel(mut slf: PyRefMut<'_, Self>, parallel: Option<bool>) -> PyRefMut<'_, Self> {
        let parallel = parallel.unwrap_or(true);
        slf.inner = std::mem::take(&mut slf.inner).with_parallel(parallel);
        slf
    }

    fn with_config(
        mut slf: PyRefMut<'_, Self>,
        config: HashMap<String, f64>,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_config(config);
        slf
    }

    /// Adjust the relative integration tolerance.
    fn with_rtol(mut slf: PyRefMut<'_, Self>, rtol: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_rtol(rtol);
        slf
    }

    /// Adjust the absolute integration tolerance.
    fn with_atol(mut slf: PyRefMut<'_, Self>, atol: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_atol(atol);
        slf
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyRefMut<'_, Self> {
        let spec = ParameterSpec::new(name.clone(), initial_value, bounds);
        slf.inner = std::mem::take(&mut slf.inner).with_parameter(spec);
        slf
    }

    /// Remove previously provided parameter defaults.
    fn clear_parameters(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.clear_parameters();
        slf
    }

    /// Select the error metric used to compare simulated and observed data.
    fn with_cost<'py>(
        mut slf: PyRefMut<'py, Self>,
        cost: PyRef<'py, PyCostMetric>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let metric = cost.metric_arc();
        slf.inner = std::mem::take(&mut slf.inner).with_cost_metric_arc(metric);
        Ok(slf)
    }

    /// Reset the cost metric to the default sum of squared errors.
    fn remove_cost(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_cost();
        slf
    }

    /// Configure the default optimiser used when `Problem.optimize` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        match &optimiser {
            Optimiser::NelderMead(nm) => {
                inner = inner.with_optimiser(nm.clone());
            }
            Optimiser::Cmaes(cma) => {
                inner = inner.with_optimiser(cma.clone());
            }
            Optimiser::Adam(adam) => {
                inner = inner.with_optimiser(adam.clone());
            }
        }
        slf.inner = inner;

        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Create a `Problem` representing the differential solver model.
    fn build(&mut self) -> PyResult<PyProblem> {
        let problem = self.inner.build().map_err(PyValueError::new_err)?;
        Ok(PyProblem {
            inner: problem,
            default_optimiser: self.default_optimiser.clone(),
        })
    }
}

// ============================================================================
// Vector Builder
// ============================================================================

/// Time-series problem builder for vector-valued objectives.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "VectorBuilder")]
pub struct PyVectorBuilder {
    inner: VectorProblemBuilder,
    default_optimiser: Option<Optimiser>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyVectorBuilder {
    /// Create an empty vector problem builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: VectorProblemBuilder::new(),
            default_optimiser: None,
        }
    }

    /// Register a callable that produces predictions matching the data shape.
    ///
    /// The callable should accept a parameter vector and return a numpy array
    /// of the same shape as the observed data.
    fn with_objective(
        mut slf: PyRefMut<'_, Self>,
        objective: Py<PyAny>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let obj_fn = move |params: &[f64]| -> Result<Vec<f64>, String> {
            Python::attach(|py| {
                let params_array = PyArray1::from_slice(py, params);
                let result = objective
                    .call1(py, (params_array,))
                    .map_err(|e| format!("Objective call failed: {}", e))?;

                let array: PyReadonlyArray1<f64> = result
                    .extract(py)
                    .map_err(|e| format!("Failed to extract array from objective: {}", e))?;

                Ok(array
                    .as_slice()
                    .map_err(|_| "Array must be contiguous")?
                    .to_vec())
            })
        };

        slf.inner = std::mem::take(&mut slf.inner).with_objective(obj_fn);
        Ok(slf)
    }

    /// Attach observed data used to fit the model.
    ///
    /// The data should be a 1D numpy array. The shape will be inferred
    /// from the data length.
    fn with_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_vec = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("Data array must be contiguous"))?
            .to_vec();

        slf.inner = std::mem::take(&mut slf.inner).with_data(data_vec);
        Ok(slf)
    }

    /// Stores an optimisation configuration value keyed by name.
    fn with_config(mut slf: PyRefMut<'_, Self>, key: String, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_config(key, value);
        slf
    }

    /// Register a named optimisation variable in the order it appears in vectors.
    #[pyo3(signature = (name, initial_value, bounds=None))]
    fn with_parameter(
        mut slf: PyRefMut<'_, Self>,
        name: String,
        initial_value: f64,
        bounds: Option<(f64, f64)>,
    ) -> PyRefMut<'_, Self> {
        let spec = ParameterSpec::new(name.clone(), initial_value, bounds);
        slf.inner = std::mem::take(&mut slf.inner).with_parameter(spec);
        slf
    }

    /// Remove previously provided parameter defaults.
    fn clear_parameters(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.clear_parameters();
        slf
    }

    /// Select the error metric used to compare predictions and observed data.
    fn with_cost<'py>(
        mut slf: PyRefMut<'py, Self>,
        cost: PyRef<'py, PyCostMetric>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let metric = cost.metric_arc();
        slf.inner = std::mem::take(&mut slf.inner).with_cost_metric_arc(metric);
        Ok(slf)
    }

    /// Reset the cost metric to the default sum of squared errors.
    fn remove_cost(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).remove_cost();
        slf
    }

    /// Configure the default optimiser used when `Problem.optimize` omits one.
    fn with_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        let mut inner = std::mem::take(&mut slf.inner);
        match &optimiser {
            Optimiser::NelderMead(nm) => {
                inner = inner.with_optimiser(nm.clone());
            }
            Optimiser::Cmaes(cma) => {
                inner = inner.with_optimiser(cma.clone());
            }
            Optimiser::Adam(adam) => {
                inner = inner.with_optimiser(adam.clone());
            }
        }
        slf.inner = inner;
        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Create a `Problem` representing the vector optimisation model.
    fn build(slf: PyRefMut<'_, Self>) -> PyResult<PyProblem> {
        let problem = slf.inner.build().map_err(PyValueError::new_err)?;
        Ok(PyProblem {
            inner: problem,
            default_optimiser: slf.default_optimiser.clone(),
        })
    }
}

// ============================================================================
// Problem
// ============================================================================

/// Executable optimisation problem wrapping the Chronopt core implementation.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: Problem,
    default_optimiser: Option<Optimiser>,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyProblem {
    /// Evaluate the configured objective function at `x`.
    fn evaluate(&self, x: Vec<f64>) -> PyResult<f64> {
        self.inner
            .evaluate(&x)
            .map_err(|e| PyValueError::new_err(format!("Evaluation failed: {}", e)))
    }

    /// Evaluate the gradient of the objective function at `x` if available.
    fn evaluate_gradient(&self, x: Vec<f64>) -> PyResult<Option<Vec<f64>>> {
        Ok(self.inner.gradient().map(|grad| grad(x.as_slice())))
    }

    #[pyo3(signature = (initial=None, optimiser=None))]
    /// Solve the problem starting from `initial` using the supplied optimiser.
    fn optimize(
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
        let result = match optimiser.as_ref().or(self.default_optimiser.as_ref()) {
            Some(Optimiser::NelderMead(nm)) => self.inner.optimize(initial, Some(nm)),
            Some(Optimiser::Cmaes(cma)) => self.inner.optimize(initial, Some(cma)),
            Some(Optimiser::Adam(adam)) => self.inner.optimize(initial, Some(adam)),
            None => self.inner.optimize(initial, None),
        };

        Ok(PyOptimisationResults { inner: result })
    }

    /// Return the numeric configuration value stored under `key` if present.
    fn get_config(&self, key: String) -> Option<f64> {
        self.inner.get_config(&key).copied()
    }

    /// Return the number of parameters the problem expects.
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn parameters(&self) -> Vec<ParameterSpecEntry> {
        self.inner
            .parameter_specs()
            .iter()
            .map(|spec| (spec.name.clone(), spec.initial_value, spec.bounds))
            .collect()
    }

    /// Return the default parameter vector implied by the builder.
    #[pyo3(name = "default_parameters")]
    fn default_parameters_py(&self) -> Vec<f64> {
        self.inner.default_parameters()
    }

    /// Return a copy of the problem configuration dictionary.
    fn config(&self) -> HashMap<String, f64> {
        self.inner.config().clone()
    }
}

// ============================================================================
// NelderMead Optimiser
// ============================================================================

/// Classic simplex-based direct search optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "NelderMead")]
#[derive(Clone)]
pub struct PyNelderMead {
    inner: NelderMead,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyNelderMead {
    /// Create a Nelder-Mead optimiser with default coefficients.
    #[new]
    fn new() -> Self {
        Self {
            inner: NelderMead::new(),
        }
    }

    /// Limit the number of simplex iterations.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on simplex size or objective reduction.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Stop once simplex vertices fall within the supplied positional tolerance.
    fn with_position_tolerance(mut slf: PyRefMut<'_, Self>, tolerance: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_position_tolerance(tolerance);
        slf
    }

    /// Abort after evaluating the objective `max_evaluations` times.
    fn with_max_evaluations(
        mut slf: PyRefMut<'_, Self>,
        max_evaluations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_evaluations(max_evaluations);
        slf
    }

    /// Override the reflection, expansion, contraction, and shrink coefficients.
    fn with_coefficients(
        mut slf: PyRefMut<'_, Self>,
        alpha: f64,
        gamma: f64,
        rho: f64,
        sigma: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_coefficients(alpha, gamma, rho, sigma);
        slf
    }

    /// Abort if the objective fails to improve within the allotted time.
    fn with_patience(mut slf: PyRefMut<'_, Self>, patience_seconds: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        slf
    }

    /// Optimise the given problem starting from the provided initial simplex centre.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

// ============================================================================
// CMAES Optimiser
// ============================================================================

/// Covariance Matrix Adaptation Evolution Strategy optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "CMAES")]
#[derive(Clone)]
pub struct PyCMAES {
    inner: CMAES,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyCMAES {
    /// Create a CMA-ES optimiser with library defaults.
    #[new]
    fn new() -> Self {
        Self {
            inner: CMAES::new(),
        }
    }

    /// Limit the number of iterations/generations before termination.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on the best objective value.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Set the initial global step-size (standard deviation).
    fn with_sigma0(mut slf: PyRefMut<'_, Self>, sigma0: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_sigma0(sigma0);
        slf
    }

    /// Abort the run if no improvement occurs for the given wall-clock duration.
    fn with_patience(mut slf: PyRefMut<'_, Self>, patience_seconds: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        slf
    }

    /// Specify the number of offspring evaluated per generation.
    fn with_population_size(
        mut slf: PyRefMut<'_, Self>,
        population_size: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_population_size(population_size);
        slf
    }

    /// Initialise the internal RNG for reproducible runs.
    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    /// Optimise the given problem starting from the provided mean vector.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

// ============================================================================
// Adam Optimiser
// ============================================================================

/// Adaptive Moment Estimation (Adam) gradient-based optimiser.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "Adam")]
#[derive(Clone)]
pub struct PyAdam {
    inner: Adam,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyAdam {
    /// Create an Adam optimiser with library defaults.
    #[new]
    fn new() -> Self {
        Self { inner: Adam::new() }
    }

    /// Limit the maximum number of optimisation iterations.
    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    /// Set the stopping threshold on the gradient norm.
    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    /// Configure the base learning rate / step size.
    fn with_step_size(mut slf: PyRefMut<'_, Self>, step_size: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_step_size(step_size);
        slf
    }

    /// Override the exponential decay rates for the first and second moments.
    fn with_betas(mut slf: PyRefMut<'_, Self>, beta1: f64, beta2: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_betas(beta1, beta2);
        slf
    }

    /// Override the numerical stability constant added to the denominator.
    fn with_eps(mut slf: PyRefMut<'_, Self>, eps: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_eps(eps);
        slf
    }

    /// Abort the run once the patience window has elapsed.
    fn with_patience(mut slf: PyRefMut<'_, Self>, patience_seconds: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        slf
    }

    /// Optimise the given problem using Adam starting from the provided point.
    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

// ============================================================================
// Optimisation Results
// ====================================================================================

/// Container for optimiser outputs and diagnostic metadata.
#[cfg_attr(feature = "stubgen", gen_stub_pyclass)]
#[pyclass(name = "OptimisationResults")]
pub struct PyOptimisationResults {
    inner: OptimisationResults,
}

#[cfg_attr(feature = "stubgen", gen_stub_pymethods)]
#[pymethods]
impl PyOptimisationResults {
    /// Decision vector corresponding to the best-found objective value.
    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    /// Objective value evaluated at `x`.
    #[getter]
    fn fun(&self) -> f64 {
        self.inner.fun
    }

    /// Number of iterations performed by the optimiser.
    #[getter]
    fn nit(&self) -> usize {
        self.inner.nit
    }

    /// Total number of objective function evaluations.
    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    /// Total number of objective function evaluations.
    #[getter]
    fn time(&self) -> Duration {
        self.inner.time
    }

    /// Whether the run satisfied its convergence criteria.
    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    /// Human-readable status message summarising the termination state.
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    /// Structured termination flag describing why the run ended.
    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination_reason.to_string()
    }

    /// Simplex vertices at termination, when provided by the optimiser.
    #[getter]
    fn final_simplex(&self) -> Vec<Vec<f64>> {
        self.inner.final_simplex.clone()
    }

    /// Objective values corresponding to `final_simplex`.
    #[getter]
    fn final_simplex_values(&self) -> Vec<f64> {
        self.inner.final_simplex_values.clone()
    }

    /// Estimated covariance of the search distribution, if available.
    #[getter]
    fn covariance(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.covariance.clone()
    }

    /// Render a concise summary of the optimisation outcome.
    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, nfev={}, time={:?}, success={}, reason={})",
            self.inner.x,
            self.inner.fun,
            self.inner.nit,
            self.inner.nfev,
            self.inner.time,
            self.inner.success,
            self.inner.message
        )
    }
}

// ============================================================================
// Stub generation helpers
// ============================================================================

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

// ============================================================================
// Module Registration
// ============================================================================

/// Return a convenience factory for creating `Builder` instances.
#[cfg_attr(feature = "stubgen", gen_stub_pyfunction)]
#[pyfunction]
fn builder_factory_py() -> PyScalarBuilder {
    PyScalarBuilder::new()
}

#[pymodule]
#[pyo3(name = "_chronopt")]
fn chronopt(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    // Register submodules for `import chronopt.builder` and `chronopt.cost`
    let sys_modules = py.import("sys")?.getattr("modules")?;
    sys_modules.set_item("chronopt.builder", &builder_module)?;
    sys_modules.set_item("chronopt.cost", &cost_module)?;
    sys_modules.set_item("chronopt.sampler", &sampler_module)?;

    // Factory function
    m.add_function(wrap_pyfunction!(builder_factory_py, m)?)?;

    Ok(())
}
