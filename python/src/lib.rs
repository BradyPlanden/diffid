use nalgebra::DMatrix;
use numpy::{Ix1, Ix2, PyArray1, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3::wrap_pyfunction;
use std::sync::Arc;

use chronopt_core::prelude::*;
use chronopt_core::problem::DiffsolBuilder;

struct PyObjectiveFn {
    callable: PyObject,
}

impl PyNelderMead {
    fn clone_inner(&self) -> NelderMead {
        self.inner.clone()
    }
}

impl PyCMAES {
    fn clone_inner(&self) -> CMAES {
        self.inner.clone()
    }
}

// Wrapper for Python callable
impl PyObjectiveFn {
    fn call(&self, x: &[f64]) -> PyResult<f64> {
        Python::with_gil(|py| {
            let callable = self.callable.bind(py);
            let input = PyArray1::from_slice(py, x);
            let result = callable.call1((input,))?;

            if let Ok(output) = result.extract::<PyReadonlyArrayDyn<f64>>() {
                let slice = output.as_slice().map_err(|_| {
                    PyValueError::new_err(
                        "Objective array must be contiguous and convertible to a slice",
                    )
                })?;

                if slice.len() == 1 {
                    return Ok(slice[0]);
                }

                return Err(PyValueError::new_err(format!(
                    "Objective array must contain exactly one element, got {}",
                    slice.len()
                )));
            }

            if let Ok(values) = result.extract::<Vec<f64>>() {
                match values.as_slice() {
                    [value] => return Ok(*value),
                    other => {
                        return Err(PyValueError::new_err(format!(
                            "Objective sequence must contain exactly one element, got {}",
                            other.len()
                        )));
                    }
                }
            }

            if let Ok(value) = result.extract::<f64>() {
                return Ok(value);
            }

            let ty_name = result
                .get_type()
                .name()
                .ok()
                .and_then(|name| name.to_str().ok().map(str::to_owned))
                .unwrap_or_else(|| "unknown".to_string());
            Err(PyTypeError::new_err(format!(
                "Objective callable must return a float, a numpy.ndarray of dtype float64, or a single-element sequence; got {}",
                ty_name
            )))
        })
    }
}

#[pyclass(name = "Builder")]
pub struct PyBuilder {
    inner: Builder,
    py_callable: Option<Arc<PyObjectiveFn>>,
    default_nm: Option<PyNelderMead>,
    default_cmaes: Option<PyCMAES>,
}

#[pymethods]
impl PyBuilder {
    #[new]
    fn new() -> Self {
        Self {
            inner: Builder::new(),
            py_callable: None,
            default_nm: None,
            default_cmaes: None,
        }
    }

    #[pyo3(name = "set_optimiser")]
    fn set_optimiser<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
        optimiser: PyObject,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let bound = optimiser.bind(py);

        if let Ok(opt) = bound.downcast::<PyNelderMead>() {
            let nm = {
                let borrow = opt.borrow();
                borrow.clone_inner()
            };

            {
                let builder = &mut *slf;
                builder.inner = std::mem::take(&mut builder.inner).set_optimiser_nm(nm.clone());
                builder.default_nm = Some(PyNelderMead { inner: nm });
                builder.default_cmaes = None;
            }

            Ok(slf)
        } else if let Ok(opt) = bound.downcast::<PyCMAES>() {
            let cma = {
                let borrow = opt.borrow();
                borrow.clone_inner()
            };

            {
                let builder = &mut *slf;
                builder.inner = std::mem::take(&mut builder.inner).set_optimiser_cmaes(cma.clone());
                builder.default_cmaes = Some(PyCMAES { inner: cma });
                builder.default_nm = None;
            }

            Ok(slf)
        } else {
            Err(PyTypeError::new_err(
                "Optimiser must be an instance of NelderMead or CMAES",
            ))
        }
    }

    fn add_callable<'a>(
        mut slf: PyRefMut<'a, Self>,
        obj: PyObject,
        py: Python<'_>,
    ) -> PyResult<PyRefMut<'a, Self>> {
        if !obj.bind(py).is_callable() {
            return Err(PyTypeError::new_err("Object must be callable"));
        }

        let py_fn = Arc::new(PyObjectiveFn { callable: obj });
        let py_fn_clone = Arc::clone(&py_fn);

        slf.inner = std::mem::take(&mut slf.inner)
            .with_objective(move |x: &[f64]| py_fn_clone.call(x).unwrap_or(f64::INFINITY));

        slf.py_callable = Some(py_fn);
        Ok(slf)
    }

    fn with_config(mut slf: PyRefMut<'_, Self>, key: String, value: f64) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).with_config(key, value);
        slf
    }

    fn add_parameter(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_parameter(name);
        slf
    }

    fn build(&mut self) -> PyResult<PyProblem> {
        let inner = std::mem::take(&mut self.inner);
        let default_nm = std::mem::take(&mut self.default_nm);
        let default_cmaes = std::mem::take(&mut self.default_cmaes);
        match inner.build() {
            Ok(problem) => Ok(PyProblem {
                inner: problem,
                default_nm,
                default_cmaes,
            }),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
}

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
        let data_matrix = match data.ndim() {
            1 => {
                let array = data
                    .as_array()
                    .into_dimensionality::<Ix1>()
                    .map_err(|_| PyValueError::new_err("Data array must be 1D or 2D"))?;
                let nrows = array.len();
                let mut column_major = Vec::with_capacity(nrows);
                column_major.extend(array.iter().copied());
                DMatrix::from_vec(nrows, 1, column_major)
            }
            2 => {
                let array = data
                    .as_array()
                    .into_dimensionality::<Ix2>()
                    .map_err(|_| PyValueError::new_err("Data array must be 1D or 2D"))?;
                let (nrows, ncols) = array.dim();
                let mut column_major = Vec::with_capacity(nrows * ncols);
                for j in 0..ncols {
                    for i in 0..nrows {
                        column_major.push(array[(i, j)]);
                    }
                }
                DMatrix::from_vec(nrows, ncols, column_major)
            }
            _ => return Err(PyValueError::new_err("Data array must be 1D or 2D")),
        };

        slf.inner = std::mem::take(&mut slf.inner).add_data(data_matrix);
        Ok(slf)
    }

    fn add_config(
        mut slf: PyRefMut<'_, Self>,
        config: std::collections::HashMap<String, f64>,
    ) -> PyRefMut<'_, Self> {
        slf.inner = std::mem::take(&mut slf.inner).add_config(config);
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
        match inner.build() {
            Ok(problem) => Ok(PyProblem {
                inner: problem,
                default_nm: None,
                default_cmaes: None,
            }),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
}

#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: Problem,
    default_nm: Option<PyNelderMead>,
    default_cmaes: Option<PyCMAES>,
}

impl PyProblem {
    pub fn get_config(&self, key: String) -> Option<f64> {
        self.inner.get_config(&key).copied()
    }

    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

#[pyclass(name = "NelderMead")]
pub struct PyNelderMead {
    inner: NelderMead,
}

#[pyclass(name = "CMAES")]
pub struct PyCMAES {
    inner: CMAES,
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

#[pymethods]
impl PyProblem {
    fn evaluate(&self, x: Vec<f64>) -> f64 {
        self.inner.evaluate(&x).unwrap()
    }

    #[pyo3(signature = (initial = None, optimiser = None))]
    fn optimize(
        &self,
        py: Python<'_>,
        initial: Option<Vec<f64>>,
        optimiser: Option<PyObject>,
    ) -> PyResult<PyOptimisationResults> {
        let result = match optimiser {
            Some(obj) => {
                let bound = obj.bind(py);
                if let Ok(opt) = bound.downcast::<PyNelderMead>() {
                    let borrow = opt.borrow();
                    self.inner.optimize(initial, Some(&borrow.inner))
                } else if let Ok(opt) = bound.downcast::<PyCMAES>() {
                    let borrow = opt.borrow();
                    self.inner.optimize(initial, Some(&borrow.inner))
                } else {
                    return Err(PyTypeError::new_err(
                        "Optimiser must be an instance of NelderMead or CMAES",
                    ));
                }
            }
            None => {
                if let Some(ref default_opt) = self.default_nm {
                    self.inner.optimize(initial, Some(&default_opt.inner))
                } else if let Some(ref default_opt) = self.default_cmaes {
                    self.inner.optimize(initial, Some(&default_opt.inner))
                } else {
                    self.inner.optimize(initial, None)
                }
            }
        };
        Ok(PyOptimisationResults { inner: result })
    }
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

#[pyfunction]
fn builder_factory_py() -> PyBuilder {
    PyBuilder::new()
}

#[pymodule]
fn chronopt(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBuilder>()?;
    m.add_class::<PyProblem>()?;
    m.add_class::<PyNelderMead>()?;
    m.add_class::<PyCMAES>()?;
    m.add_class::<PyOptimisationResults>()?;
    m.add_class::<PyDiffsolBuilder>()?;

    // Add alias: PythonBuilder -> Builder (type alias at module level)
    let builder_type = PyType::new::<PyBuilder>(py);
    let builder_type_owned = builder_type.unbind();
    m.add("PythonBuilder", builder_type_owned)?;

    // Create builder submodule
    let builder_module = PyModule::new(py, "builder")?;
    builder_module.add_class::<PyDiffsolBuilder>()?;
    m.add_submodule(&builder_module)?;

    // Also add Diffsol directly to the main module for convenience
    m.add_class::<PyDiffsolBuilder>()?;

    // Provide a simple factory function for convenience
    m.add_function(wrap_pyfunction!(builder_factory_py, py)?)?;
    Ok(())
}
