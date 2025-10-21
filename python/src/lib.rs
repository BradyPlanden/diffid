use nalgebra::DMatrix;
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

// Wrapper for Python callable
impl PyObjectiveFn {
    fn call(&self, x: &[f64]) -> PyResult<f64> {
        Python::with_gil(|py| {
            let callable = self.callable.bind(py);
            callable.call1((x.to_vec(),))?.extract()
        })
    }
}

#[pyclass(name = "Builder")]
pub struct PyBuilder {
    inner: Builder,
    py_callable: Option<Arc<PyObjectiveFn>>,
    default_optimiser: Option<PyNelderMead>,
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

    fn set_optimiser<'a>(
        mut slf: PyRefMut<'a, Self>,
        optimiser: PyRef<'a, PyNelderMead>,
    ) -> PyRefMut<'a, Self> {
        slf.inner = std::mem::take(&mut slf.inner).set_optimiser(optimiser.inner.clone());
        slf.default_optimiser = Some(PyNelderMead {
            inner: optimiser.inner.clone(),
        });
        slf
    }

    fn build(&mut self) -> PyResult<PyProblem> {
        let inner = std::mem::take(&mut self.inner);
        let default_optimiser = std::mem::take(&mut self.default_optimiser);
        match inner.build() {
            Ok(problem) => Ok(PyProblem {
                inner: problem,
                default_optimiser,
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

    fn add_data(mut slf: PyRefMut<'_, Self>, data: Vec<f64>) -> PyRefMut<'_, Self> {
        let ncols = data.len();
        let data_matrix = DMatrix::from_vec(ncols, 1, data);
        slf.inner = std::mem::take(&mut slf.inner).add_data(data_matrix);
        slf
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
                default_optimiser: None,
            }),
            Err(e) => Err(PyValueError::new_err(e)),
        }
    }
}

#[pyclass(name = "Problem")]
pub struct PyProblem {
    inner: Problem,
    default_optimiser: Option<PyNelderMead>,
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
    fn success(&self) -> bool {
        self.inner.success
    }

    fn __repr__(&self) -> String {
        format!(
            "OptimisationResults(x={:?}, fun={:.6}, nit={}, success={})",
            self.inner.x, self.inner.fun, self.inner.nit, self.inner.success
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
        initial: Option<Vec<f64>>,
        optimiser: Option<&PyNelderMead>,
    ) -> PyOptimisationResults {
        let result = match optimiser {
            Some(opt) => self.inner.optimize(initial, Some(&opt.inner)),
            None => {
                // Use the default optimizer if available, otherwise let the inner Problem handle it
                if let Some(ref default_opt) = self.default_optimiser {
                    self.inner.optimize(initial, Some(&default_opt.inner))
                } else {
                    self.inner.optimize(initial, None)
                }
            }
        };
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
