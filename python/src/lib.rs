use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use std::sync::Arc;

use chronopt_core::prelude::*;
use chronopt_core::problem::DiffsolBuilder;

// ============================================================================
// Optimiser Enum for Polymorphic Types
// ============================================================================

#[derive(Clone)]
enum Optimiser {
    NelderMead(NelderMead),
    CMAES(CMAES),
}

impl<'py> FromPyObject<'py> for Optimiser {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(nm) = ob.extract::<PyRef<PyNelderMead>>() {
            Ok(Optimiser::NelderMead(nm.inner.clone()))
        } else if let Ok(cma) = ob.extract::<PyRef<PyCMAES>>() {
            Ok(Optimiser::CMAES(cma.inner.clone()))
        } else {
            Err(PyTypeError::new_err(
                "Optimiser must be an instance of NelderMead or CMAES"
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
        Python::with_gil(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            // Try extracting as array first
            if let Ok(output) = result.extract::<PyReadonlyArray1<f64>>() {
                let array = output.as_array();
                return match array.len() {
                    1 => Ok(array[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective array must contain exactly one element, got {}", n
                    ))),
                };
            }

            // Try extracting as sequence
            if let Ok(values) = result.extract::<Vec<f64>>() {
                return match values.len() {
                    1 => Ok(values[0]),
                    n => Err(PyValueError::new_err(format!(
                        "Objective sequence must contain exactly one element, got {}", n
                    ))),
                };
            }

            // Try extracting as scalar
            if let Ok(value) = result.extract::<f64>() {
                return Ok(value);
            }

            // Error case
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

// ============================================================================
// Builder
// ============================================================================

#[pyclass(name = "Builder")]
pub struct PyBuilder {
    inner: Builder,
    py_callable: Option<Arc<PyObjectiveFn>>,
    default_optimiser: Option<Optimiser>,
}

#[pymethods]
impl PyBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Builder::new(),
            py_callable: None,
            default_optimiser: None,
        }
    }

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

    fn add_callable(mut slf: PyRefMut<'_, Self>, obj: Py<PyAny>) -> PyResult<PyRefMut<'_, Self>> {
        Python::with_gil(|py| {
            if !obj.bind(py).is_callable() {
                return Err(PyTypeError::new_err("Object must be callable"));
            }
            Ok(())
        })?;

        let py_fn = Arc::new(PyObjectiveFn::new(obj));
        let py_fn_clone = Arc::clone(&py_fn);

        slf.inner = std::mem::take(&mut slf.inner)
            .with_objective(move |x: &[f64]| {
                py_fn_clone.call(x).unwrap_or(f64::INFINITY)
            });

        slf.py_callable = Some(py_fn);
        Ok(slf)
    }

    fn add_parameter(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_parameter(name);
        slf
    }

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
// DiffsolBuilder
// ============================================================================

#[pyclass(name = "DiffsolBuilder")]
pub struct PyDiffsolBuilder {
    inner: DiffsolBuilder,
}

#[pymethods]
impl PyDiffsolBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: DiffsolBuilder::new(),
        }
    }

    fn add_diffsl(mut slf: PyRefMut<'_, Self>, dsl: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_diffsl(dsl);
        slf
    }

    fn add_data<'py>(
        mut slf: PyRefMut<'py, Self>,
        data: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let data_matrix = convert_array_to_dmatrix(&data)?;
        slf.inner = std::mem::take(&mut slf.inner).add_data(data_matrix);
        Ok(slf)
    }

    fn with_t_span(
        mut slf: PyRefMut<'_, Self>,
        t_span: Vec<f64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        if t_span.is_empty() {
            return Err(PyValueError::new_err("t_span must not be empty"));
        }
        slf.inner = std::mem::take(&mut slf.inner).with_t_span(t_span);
        Ok(slf)
    }

    fn with_rtol(mut slf: PyRefMut<'_, Self>, rtol: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_rtol(rtol);
        slf
    }

    fn with_atol(mut slf: PyRefMut<'_, Self>, atol: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_atol(atol);
        slf
    }

    fn add_params(
        mut slf: PyRefMut<'_, Self>,
        params: std::collections::HashMap<String, f64>,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_params(params);
        slf
    }

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
            let slice = data.as_slice()
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

#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: Problem,
    default_optimiser: Option<Optimiser>,
}

#[pymethods]
impl PyProblem {
    fn evaluate(&self, x: Vec<f64>) -> PyResult<f64> {
        self.inner.evaluate(&x)
            .map_err(|e| PyValueError::new_err(format!("Evaluation failed: {}", e)))
    }

    #[pyo3(signature = (initial=None, optimiser=None))]
    fn optimize(
        &self,
        initial: Option<Vec<f64>>,
        optimiser: Option<Optimiser>,
    ) -> PyResult<PyOptimisationResults> {
        let opt = optimiser.as_ref()
            .or(self.default_optimiser.as_ref());

        let result = match opt {
            Some(Optimiser::NelderMead(nm)) => {
                self.inner.optimize(initial, Some(nm))
            }
            Some(Optimiser::CMAES(cma)) => {
                self.inner.optimize(initial, Some(cma))
            }
            None => self.inner.optimize(initial, None),
        };

        Ok(PyOptimisationResults { inner: result })
    }

    fn get_config(&self, key: String) -> Option<f64> {
        self.inner.get_config(&key).copied()
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

// ============================================================================
// NelderMead Optimiser
// ============================================================================

#[pyclass(name = "NelderMead")]
#[derive(Clone)]
pub struct PyNelderMead {
    inner: NelderMead,
}

#[pymethods]
impl PyNelderMead {
    #[new]
    fn new() -> Self {
        Self {
            inner: NelderMead::new(),
        }
    }

    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    fn with_position_tolerance(mut slf: PyRefMut<'_, Self>, tolerance: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_position_tolerance(tolerance);
        slf
    }

    fn with_max_evaluations(
        mut slf: PyRefMut<'_, Self>,
        max_evaluations: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_evaluations(max_evaluations);
        slf
    }

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

    fn with_patience(mut slf: PyRefMut<'_, Self>, patience_seconds: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        slf
    }

    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

// ============================================================================
// CMAES Optimiser
// ============================================================================

#[pyclass(name = "CMAES")]
#[derive(Clone)]
pub struct PyCMAES {
    inner: CMAES,
}

#[pymethods]
impl PyCMAES {
    #[new]
    fn new() -> Self {
        Self {
            inner: CMAES::new(),
        }
    }

    fn with_max_iter(mut slf: PyRefMut<'_, Self>, max_iter: usize) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_max_iter(max_iter);
        slf
    }

    fn with_threshold(mut slf: PyRefMut<'_, Self>, threshold: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_threshold(threshold);
        slf
    }

    fn with_sigma0(mut slf: PyRefMut<'_, Self>, sigma0: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_sigma0(sigma0);
        slf
    }

    fn with_patience(mut slf: PyRefMut<'_, Self>, patience_seconds: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_patience(patience_seconds);
        slf
    }

    fn with_population_size(
        mut slf: PyRefMut<'_, Self>,
        population_size: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_population_size(population_size);
        slf
    }

    fn with_seed(mut slf: PyRefMut<'_, Self>, seed: u64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_seed(seed);
        slf
    }

    fn run(&self, problem: &PyProblem, initial: Vec<f64>) -> PyOptimisationResults {
        let result = self.inner.run(&problem.inner, initial);
        PyOptimisationResults { inner: result }
    }
}

// ============================================================================
// Optimisation Results
// ============================================================================

#[pyclass(name = "OptimisationResults")]
pub struct PyOptimisationResults {
    inner: OptimisationResults,
}

#[pymethods]
impl PyOptimisationResults {
    #[getter]
    fn x(&self) -> Vec<f64> {
        self.inner.x.clone()
    }

    #[getter]
    fn fun(&self) -> f64 {
        self.inner.fun
    }

    #[getter]
    fn nit(&self) -> usize {
        self.inner.nit
    }

    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    #[getter]
    fn success(&self) -> bool {
        self.inner.success
    }

    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    #[getter]
    fn termination_reason(&self) -> String {
        self.inner.termination_reason.to_string()
    }

    #[getter]
    fn final_simplex(&self) -> Vec<Vec<f64>> {
        self.inner.final_simplex.clone()
    }

    #[getter]
    fn final_simplex_values(&self) -> Vec<f64> {
        self.inner.final_simplex_values.clone()
    }

    #[getter]
    fn covariance(&self) -> Option<Vec<Vec<f64>>> {
        self.inner.covariance.clone()
    }

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