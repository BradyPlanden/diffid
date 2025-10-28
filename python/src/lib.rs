use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;

use chronopt_core::prelude::*;
use chronopt_core::problem::{Builder, DiffsolBackend, DiffsolBuilder};

#[cfg(feature = "stubgen")]
use pyo3_stub_gen::{define_stub_info_gatherer, TypeInfo};

#[cfg(all(feature = "stubgen", feature = "extension-module"))]
compile_error!(
    "The 'stubgen' feature must be built without the 'extension-module' feature. \
     Run with `--no-default-features --features stubgen`."
);

// ============================================================================
// Optimiser Enum for Polymorphic Types
// ============================================================================

#[derive(Clone)]
enum Optimiser {
    NelderMead(NelderMead),
    CMAES(CMAES),
}

#[cfg(feature = "stubgen")]
#[allow(dead_code)]
fn optimiser_type_info() -> TypeInfo {
    TypeInfo::unqualified("chronopt.NelderMead") | TypeInfo::unqualified("chronopt.CMAES")
}

impl<'py> FromPyObject<'py> for Optimiser {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(nm) = ob.extract::<PyRef<PyNelderMead>>() {
            Ok(Optimiser::NelderMead(nm.inner.clone()))
        } else if let Ok(cma) = ob.extract::<PyRef<PyCMAES>>() {
            Ok(Optimiser::CMAES(cma.inner.clone()))
        } else {
            Err(PyTypeError::new_err(
                "Optimiser must be an instance of NelderMead or CMAES",
            ))
        }
    }
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
#[pyclass(name = "Builder")]
pub struct PyBuilder {
    inner: Builder,
    py_callable: Option<Arc<PyObjectiveFn>>,
    py_gradient: Option<Arc<PyGradientFn>>,
    default_optimiser: Option<Optimiser>,
}

#[pymethods]
impl PyBuilder {
    /// Create an empty builder with no objective, parameters, or default optimiser.
    #[new]
    fn new() -> Self {
        Self {
            inner: Builder::new(),
            py_callable: None,
            py_gradient: None,
            default_optimiser: None,
        }
    }

    /// Configure the default optimiser used when `Problem.optimize` omits one.
    #[pyo3(name = "set_optimiser")]
    fn set_optimiser(mut slf: PyRefMut<'_, Self>, optimiser: Optimiser) -> PyRefMut<'_, Self> {
        slf.inner = match &optimiser {
            Optimiser::NelderMead(nm) => {
                std::mem::take(&mut slf.inner).set_optimiser_nm(nm.clone())
            }
            Optimiser::CMAES(cma) => {
                std::mem::take(&mut slf.inner).set_optimiser_cmaes(cma.clone())
            }
        };

        slf.default_optimiser = Some(optimiser);
        slf
    }

    /// Attach the objective function callable executed during optimisation.
    fn add_callable(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
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

    #[pyo3(name = "add_gradient")]
    fn add_gradient(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
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
    fn add_parameter(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_parameter(name);
        slf
    }

    /// Finalize the builder into an executable `Problem`.
    fn build(&mut self) -> PyResult<PyProblem> {
        let inner = std::mem::take(&mut self.inner);
        let default_optimiser = self.default_optimiser.take();
        let problem = inner.build().map_err(|e| PyValueError::new_err(e))?;
        Ok(PyProblem {
            inner: problem,
            default_optimiser,
        })
    }
}

// ============================================================================
// Diffsol Builder
// ============================================================================

/// Differential equation solver builder.
#[pyclass(name = "DiffsolBuilder")]
pub struct PyDiffsolBuilder {
    inner: DiffsolBuilder,
}

#[pymethods]
impl PyDiffsolBuilder {
    /// Create an empty differential solver builder.
    #[new]
    fn new() -> Self {
        Self {
            inner: DiffsolBuilder::new(),
        }
    }

    /// Register the DiffSL program describing the system dynamics.
    fn add_diffsl(mut slf: PyRefMut<'_, Self>, dsl: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_diffsl(dsl);
        slf
    }

    /// Attach observed data used to fit the differential equation.
    fn add_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_matrix = convert_array_to_dmatrix(&data)?;
        slf.inner = std::mem::take(&mut slf.inner).add_data(data_matrix);
        Ok(slf)
    }

    /// Set the time sampling points or integration window.
    fn with_t_span(mut slf: PyRefMut<'_, Self>, t_span: Vec<f64>) -> PyResult<PyRefMut<'_, Self>> {
        if t_span.is_empty() {
            return Err(PyValueError::new_err("t_span must not be empty"));
        }
        slf.inner = std::mem::take(&mut slf.inner).with_t_span(t_span);
        Ok(slf)
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

    /// Provide named parameter defaults for the DiffSL program.
    fn add_params(
        mut slf: PyRefMut<'_, Self>,
        params: std::collections::HashMap<String, f64>,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_params(params);
        slf
    }

    /// Create a `Problem` representing the differential solver model.
    fn build(&mut self) -> PyResult<PyProblem> {
        let inner = std::mem::take(&mut self.inner);
        let problem = inner.build().map_err(|e| PyValueError::new_err(e))?;

        Ok(PyProblem {
            inner: problem,
            default_optimiser: None,
        })
    }
}

// Helper function to convert numpy arrays to DMatrix
fn convert_array_to_dmatrix(data: &PyReadonlyArrayDyn<f64>) -> PyResult<DMatrix<f64>> {
    match data.ndim() {
        1 => {
            let slice = data
                .as_slice()
                .map_err(|_| PyValueError::new_err("Array must be contiguous"))?;
            Ok(DMatrix::from_vec(slice.len(), 1, slice.to_vec()))
        }
        2 => {
            // Extract shape information
            let shape = data.shape();
            let (nrows, ncols) = (shape[0], shape[1]);

            // Try to get as 2D array for efficient access
            if let Ok(array2d) = data.as_array().into_dimensionality::<numpy::Ix2>() {
                let mut column_major = Vec::with_capacity(nrows * ncols);
                for j in 0..ncols {
                    for i in 0..nrows {
                        column_major.push(array2d[[i, j]]);
                    }
                }
                Ok(DMatrix::from_vec(nrows, ncols, column_major))
            } else {
                Err(PyValueError::new_err("Failed to convert 2D array"))
            }
        }
        _ => Err(PyValueError::new_err("Data array must be 1D or 2D")),
    }
}

// ============================================================================
// Problem
// ============================================================================

/// Executable optimisation problem wrapping the Chronopt core implementation.
#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: Problem,
    default_optimiser: Option<Optimiser>,
}

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
        let opt = optimiser.as_ref().or(self.default_optimiser.as_ref());

        let result = match opt {
            Some(Optimiser::NelderMead(nm)) => self.inner.optimize(initial, Some(nm)),
            Some(Optimiser::CMAES(cma)) => self.inner.optimize(initial, Some(cma)),
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
}

// ============================================================================
// NelderMead Optimiser
// ============================================================================

/// Classic simplex-based direct search optimiser.
#[pyclass(name = "NelderMead")]
#[derive(Clone)]
pub struct PyNelderMead {
    inner: NelderMead,
}

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
#[pyclass(name = "CMAES")]
#[derive(Clone)]
pub struct PyCMAES {
    inner: CMAES,
}

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
// Optimisation Results
// ============================================================================

/// Container for optimiser outputs and diagnostic metadata.
#[pyclass(name = "OptimisationResults")]
pub struct PyOptimisationResults {
    inner: OptimisationResults,
}

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
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, nfev={}, success={}, reason={})",
            self.inner.x,
            self.inner.fun,
            self.inner.nit,
            self.inner.nfev,
            self.inner.success,
            self.inner.message
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

#[cfg(feature = "stubgen")]
define_stub_info_gatherer!(stub_info);

/// Return a convenience factory for creating `Builder` instances.
#[pyfunction]
fn builder_factory_py() -> PyBuilder {
    PyBuilder::new()
}

#[pymodule]
fn chronopt(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Main classes
    m.add_class::<PyBuilder>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyNelderMead>()?;
    m.add_class::<PyCMAES>()?;
    m.add_class::<PyOptimisationResults>()?;
    m.add_class::<PyDiffsolBuilder>()?;

    // Alias for backwards compatibility
    let builder_type = PyType::new::<PyBuilder>(py);
    let builder_type_owned = builder_type.unbind();
    m.add("PythonBuilder", builder_type_owned)?;

    // Builder submodule
    let builder_module = PyModule::new(py, "builder")?;
    builder_module.add_class::<PyDiffsolBuilder>()?;
    m.add_submodule(&builder_module)?;

    // Factory function
    m.add_function(wrap_pyfunction!(builder_factory_py, m)?)?;

    Ok(())
}
